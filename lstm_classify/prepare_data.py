import os
from tqdm import tqdm


class PrepareData(object):
    def __init__(self):
        self.base_dir = os.path.join(os.path.dirname(__file__), "raw_data")
        self.raw_data_path = os.path.join(self.base_dir, "toutiao_cat_data.txt")
        self.prepared_data_path = os.path.join(self.base_dir, "toutiao_prepared.txt")

    def obtain_raw_data(self):
        """"""
        with open(self.raw_data_path, "r", encoding="utf8") as reader:
            all_lines = reader.readlines()
        prepared_data = []
        print("正在处理数据...")
        for line in tqdm(all_lines):
            info = self.deal_data(line)
            if info:
                prepared_data.append(info)
        # 保存处理好的数据
        with open(self.prepared_data_path, "w", encoding="utf8") as writer:
            for info in prepared_data:
                # print(info)
                writer.write(info + "\n")

    @staticmethod
    def deal_data(line):
        """"""
        line_split = line.split("_!_")
        label_name = line_split[2]
        content = line_split[3]
        desc = line_split[4]

        text = content + " " + desc
        text = text.replace("\t", " ")
        text = text.replace("\n", " ")

        if text and label_name:
            return text + "\t" + label_name
        else:
            return None


if __name__ == '__main__':
    prepared_obj = PrepareData()
    prepared_obj.obtain_raw_data()
