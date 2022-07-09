import logging
import os
import time
from functools import wraps

from tools.dataframe import is_exist_df, save_df, load_df


def logging_run_info(log_input=True, log_input_keys=None, log_output=False, log_level=logging.DEBUG):
    """
    :param log_input:是否打印input
    :param log_input_keys:字符串数组，指明打印哪些kwargs中的参数；默认None表示打印全部input
    :param log_output:是否打印output
    :param log_level:日志级别，默认DEBUG
    :return:
    """

    def _wraper(func):
        _func = func
        while hasattr(_func, '__wrapped__'):
            _func = _func.__wrapped__
        _varnames_ = _func.__code__.co_varnames[:_func.__code__.co_argcount]
        _defaults_ = _func.__defaults__ or ()
        _param_defaults_ = dict(zip(_varnames_[-len(_defaults_):], _defaults_))
        _log_input_keys = log_input_keys or _varnames_

        @wraps(func)
        def decorated(*args, **kwargs):
            func_params = {}
            for i, varname in enumerate(_varnames_):
                if i < len(args):
                    func_params[varname] = args[i]
                elif varname in kwargs:
                    func_params[varname] = kwargs[varname]
                elif varname in _param_defaults_:
                    func_params[varname] = _param_defaults_[varname]
                else:
                    raise ValueError(f"function:{func.__name__}{_varnames_} missing arguments:{varname}")
            if log_input:
                _input_log_str = ', '.join(map(lambda k: f"{k}={func_params[k]}", _log_input_keys))
            else:
                _input_log_str = ''
            logging.log(log_level, f"start func: {func.__name__}({_input_log_str})")
            t1 = time.time()
            ans = func(*args, **kwargs)
            t2 = time.time()
            logging.log(log_level, f"finish func: {func.__name__}, time cost: {round(t2 - t1, 3)}s")
            if log_output:
                logging.log(log_level, f"output is : {ans}")
            return ans

        return decorated

    return _wraper


def cache_output_df(cache_root_dir):
    """
    将函数返回的dataframe缓存到本地，下次传入相同参数时，跳过函数直接从本地缓存读取
    * 如果和其他装饰器连用，则该装饰器最好放在最下面
    * 如果要重建缓存文件，需要设置环境变量: REBUILD_CACHE="true"
    :param cache_root_dir:缓存文件的根路径，缓存文件会放在"{cache_root_dir}/{func_name}/param1=value1&param2=value2.parquet.snappy"
                          如果函数是无参的，则文件会放在"{cache_root_dir}/{func_name}.parquet.snappy"
    :return:dataframe
    """

    def _wraper(func):
        _func = func
        while hasattr(_func, '__wrapped__'):
            _func = _func.__wrapped__
        _varnames_ = _func.__code__.co_varnames[:_func.__code__.co_argcount]
        _defaults_ = _func.__defaults__ or ()
        _param_defaults_ = dict(zip(_varnames_[-len(_defaults_):], _defaults_))

        @wraps(func)
        def decorated(*args, **kwargs):
            func_params = {}
            for i, varname in enumerate(_varnames_):
                if i < len(args):
                    func_params[varname] = args[i]
                elif varname in kwargs:
                    func_params[varname] = kwargs[varname]
                elif varname in _param_defaults_:
                    func_params[varname] = _param_defaults_[varname]
                else:
                    raise ValueError(f"function:{func.__name__}{_varnames_} missing arguments:{varname}")
            _cache_file_path = os.path.join(cache_root_dir, func.__name__)
            if func_params:
                _file_name = '&'.join(map(lambda kv: f"{kv[0]}={_to_str_(kv[1])}", func_params.items()))
                _cache_file_path = os.path.join(_cache_file_path, _file_name)
            _cache_file_path = os.path.abspath(_cache_file_path)

            if os.environ.get("REBUILD_CACHE", "false") == "true":
                logging.debug(f"rebuild cached file : {_cache_file_path}")
                _df = func(*args, **kwargs)
                save_df(_df, _cache_file_path)
            elif is_exist_df(_cache_file_path):
                logging.debug(f"load output df from cached file : {_cache_file_path}")
                _df = load_df(_cache_file_path)
            else:
                _df = func(*args, **kwargs)
                save_df(_df, _cache_file_path)
            return _df

        return decorated

    return _wraper


def _to_str_(obj):
    if isinstance(obj, (int, str)):
        return str(obj)
    elif isinstance(obj, (list, range, map)):
        return ",".join(map(str, obj))
    elif isinstance(obj, (dict,)):
        return ','.join(map(lambda kv: f"{kv[0]}_{kv[1]}", obj.items()))
    else:
        raise Exception(f"not support such input obj type : {type(obj)}")
