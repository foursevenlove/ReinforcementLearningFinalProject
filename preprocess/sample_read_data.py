import csv
import json

from dateutil.parser import parse

from constants import *

worker_quality = {}
csvfile = open(f"{ROOT_DATA_PATH}/worker_quality.csv", "r")
csvreader = csv.reader(csvfile)
print(f"csv head : {','.join(next(csvreader))}")
for line in csvreader:
    if float(line[1]) > 0.0:
        worker_quality[int(line[0])] = float(line[1]) / 100.0
csvfile.close()

file = open(f"{ROOT_DATA_PATH}/project_list.csv", "r")
project_list_lines = file.readlines()
file.close()
project_dir = f"{ROOT_DATA_PATH}/project/"
entry_dir = f"{ROOT_DATA_PATH}/entry/"

all_begin_time = parse("2018-01-01T0:0:0Z")

project_info = {}
entry_info = {}
limit = 24
industry_list = {}
for line in project_list_lines:
    line = line.strip('\n').split(',')

    project_id = int(line[0])
    entry_count = int(line[1])

    file = open(project_dir + "project_" + str(project_id) + ".txt", "r", encoding="utf-8")
    htmlcode = file.read()
    file.close()
    text = json.loads(htmlcode)

    project_info[project_id] = {}
    project_info[project_id]["sub_category"] = int(text["sub_category"])  # % project sub_category
    project_info[project_id]["category"] = int(text["category"])  # % project category
    project_info[project_id]["entry_count"] = int(text["entry_count"])  # % project answers in total
    project_info[project_id]["start_date"] = parse(text["start_date"])  # % project start date
    project_info[project_id]["deadline"] = parse(text["deadline"])  # % project end date

    if text["industry"] not in industry_list:
        industry_list[text["industry"]] = len(industry_list)
    project_info[project_id]["industry"] = industry_list[text["industry"]]  # % project domain

    entry_info[project_id] = {}
    k = 0
    while k < entry_count:
        file = open(entry_dir + "entry_" + str(project_id) + "_" + str(k) + ".txt", "r", encoding="utf-8")
        htmlcode = file.read()
        file.close()
        text = json.loads(htmlcode)

        for item in text["results"]:
            entry_number = int(item["entry_number"])
            entry_info[project_id][entry_number] = {}
            entry_info[project_id][entry_number]["entry_created_at"] = parse(
                item["entry_created_at"])  # % worker answer_time
            # entry_info[project_id][entry_number]["worker"] = int(item["worker"])  # item里面没有worker字段，应该是下面的author字段
            entry_info[project_id][entry_number]["worker"] = int(item["author"])  # % work_id
        k += limit

print("finish read_data")

