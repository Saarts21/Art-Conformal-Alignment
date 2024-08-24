import pandas as pd
import numpy as np
import re

def clear_bad_rows():
    table_path = "data.csv"
    df = pd.read_csv(table_path)
    indices_to_delete = [13, 38, 146]
    df = df.drop(indices_to_delete)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(table_path, index=False)

def rename_column():
    table_path = "data.csv"
    df = pd.read_csv(table_path)
    df.rename(columns={'old name': 'new name'}, inplace=True)
    df.to_csv(table_path, index=False)

def convert_string_to_numpy_array(s):
    s = s.strip('[]')
    lst = s.split()
    return np.array(lst, dtype=float)

def count_aligned_and_correct_artist_pred():
    table_path = "data.csv"
    df = pd.read_csv(table_path)

    count = 0
    pred_count = 0
    align_count = 0
    for i, row in df.iterrows():
        b_pred = b_align = False
        ground_truth_index = np.argmax(convert_string_to_numpy_array(df['ground_truth_artist_vector'][i]))
        predicted_index = np.argmax(convert_string_to_numpy_array(df['predicted_prob_artists_vector'][i]))
        print(f"ground_truth_index = {ground_truth_index}, predicted_index = {predicted_index}")
        if ground_truth_index == predicted_index:
            b_pred = True
            pred_count += 1
        if df['alignment_score'][i] == 1:
            b_align = True
            align_count += 1
        if b_pred and b_align:
            count += 1
    print(f"aligned: {align_count}, predicted: {pred_count}, intersection: {count}")

def parse_out_objects():
    objects_list_pattern = r"generated_image_objects:\s*(\[[^\]]*\])"

    with open("out.txt", "r") as file:
        content = file.read()

    elements = content.split("row")

    i = -1
    for e in elements:
        match = re.findall(objects_list_pattern, e)
        if len(match) == 0:
            continue
        i += 1
        objects_list = eval(match[0])

        print(f"Row {i}: {objects_list}")


def take():
    with open("more.txt", "r") as file:
        content = file.readlines()
    results = []

    for line in content:
        row = re.search(r"--- row (\d+) ---", line)
        if row:
            results.append(int(row.group(1)))
        count = re.search(r"shared_elements_count:\s+([+-]?\d*\.\d+)", line)
        if count:
            results.append(float(count.group(1)))
    
    indices_of_1 = []
    for res in results:
        if res == 1.0:
            indices_of_1.append(last_res)
        last_res = res
    print(indices_of_1)
    print(len(indices_of_1))


def take_2():
    table_path = "data.csv"
    df = pd.read_csv(table_path)
    indices_of_1 = [336, 373, 431, 434, 439, 446, 471, 480, 487, 488, 502, 503, 505, 508, 516, 520,
                    542, 544, 551, 557, 560, 562, 563, 571, 578, 579, 586, 589, 590, 592, 593, 597,
                    599, 600, 604, 606, 611, 617, 619, 620, 625, 626, 631, 632, 634, 638, 639, 640,
                    642, 644, 649, 651]

    for i in indices_of_1:
        df.at[i, 'shared_elements_count'] = 1.0

    df.to_csv(table_path, index=False)

