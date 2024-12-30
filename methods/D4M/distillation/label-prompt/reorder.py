# 定义文件路径
file_path = 'cifar-100.txt'  # 替换为你的文件路径

# 读取文件中的内容
with open(file_path, 'r') as f:
    lines = f.readlines()

# 去掉每行末尾的换行符，并进行排序
lines = [line.strip() for line in lines]  # 去掉换行符
lines.sort()  # 默认按字母顺序排序，若需要按其他规则可修改

# 将排序后的内容写回文件
with open(file_path, 'r') as f:
    with open('new_order.txt', 'w') as fr:
        for line in lines:
            fr.write(line + '\n')

print("文件内容已根据字母顺序排序！")
