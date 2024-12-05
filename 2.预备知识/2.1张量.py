import torch

import torch

# 创建一个包含单个元素的张量
tensor = torch.tensor(42.0)

# 转换为标量
scalar = tensor.item()
print("从张量到标量：", scalar)
print("张量转float：", float(tensor))
print("张量转int：", int(tensor))

# # 创建一个标量
# scalar = 5.0
#
# # 转换为张量
# tensor_from_scalar = torch.tensor(scalar)
# print("从标量到张量：", tensor_from_scalar)
# print("从标量转化的张量的维度：", tensor_from_scalar.dim())
# print("从标量转化的张量的形状：", tensor_from_scalar.shape)

# 创建一个 Python 列表
# python_list = [[1, 2, 3], [4, 5, 6]]
#
# # 转换为张量
# tensor_from_list = torch.tensor(python_list)
# print("张量：\n", tensor_from_list)

# 创建一个张量
# tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
#
# # 转换为 Python 列表
# python_list = tensor.tolist()
# print("Python 列表：", python_list)

# # 创建一个 NumPy 数组
# numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
#
# # 转换为张量
# tensor_from_numpy = torch.from_numpy(numpy_array)
# print("张量：\n", tensor_from_numpy)

# 创建一个张量
# tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
#
# # 转换为 NumPy 数组
# numpy_array = tensor.numpy()
# print("NumPy 数组：\n", numpy_array)

# # 定义多个操作的函数
# def computation(X, Y):
#     Z = torch.zeros_like(Y)  # 分配一个全零的张量（未使用）
#     A = X + Y                # 计算 A
#     B = A + Y                # 计算 B
#     C = B + Y                # 计算 C
#     return C + Y            # 返回 C + Y
#
#
# X = torch.tensor([1.0, 2.0, 3.0])
# Y = torch.tensor([4.0, 5.0, 6.0])
#
# # 调用函数
# result = computation(X, Y)
# print("计算结果：", result)


# 创建两个张量
# X = torch.tensor([1.0, 2.0, 3.0])
# Y = torch.tensor([4.0, 5.0, 6.0])
#
# # 显示 Y 的内存地址
# before_id = id(Y)
# print("Y 的内存地址：", before_id)
#
# # 原地更新 Y
# Y += X  # 直接在 Y 的位置上进行更新
#
# # 检查 Y 的内存地址
# after_id = id(Y)
# print("更新后 Y 的内存地址：", after_id)
#
# # 检查地址是否相同
# print("地址相同吗？", before_id == after_id)  # 输出：True

# 创建两个张量
# X = torch.tensor([1.0, 2.0, 3.0])
# Y = torch.tensor([4.0, 5.0, 6.0])
#
# # 显示 Y 的内存地址
# before_id = id(Y)
# print("Y 的内存地址：", before_id)
#
# # 执行操作 Y = Y + X
# Y = Y + X  # 这里会创建一个新张量并更新 Y 的引用
#
# # 检查 Y 的内存地址
# after_id = id(Y)
# print("更新后 Y 的内存地址：", after_id)
#
# # 检查地址是否相同
# print("地址相同吗？", before_id == after_id)  # 输出：False

# 创建一个 2x3x4 的张量
# tensor = torch.tensor([[[1, 2, 3, 4],
#                         [5, 6, 7, 8],
#                         [9, 10, 11, 12]],
#
#                        [[13, 14, 15, 16],
#                         [17, 18, 19, 20],
#                         [21, 22, 23, 24]]])

# 创建一个布尔索引，选择大于 10 的元素
# bool_index = tensor > 10
#
# print("布尔索引张量：\n", bool_index, end='\n\n')
#
# filtered_tensor = tensor[bool_index]
# print("过滤出大于 10 的元素：\n", filtered_tensor)

# 修改第 2 个图像的第 3 行第 4 列的元素
# tensor[1, 2, 3] = 99
# print("修改后的张量：\n", tensor)

# 获取第 1 个图像的前 2 行和所有列
# combined_slice = tensor[0, :2, :]
# print("获取第 1 个图像的前 2 行和所有列：\n", combined_slice)

# 获取第 1 个和第 2 个图像的第 2 行的所有列
# advanced_index = tensor[[0, 1], 1, :]
# print("获取第 1 个和第 2 个图像的第 2 行的所有列：\n", advanced_index)

# 获取所有图像的第 1 行和第 2 列
# slice_2 = tensor[:, 0, 1]
# print("获取所有图像的第 1 行和第 2 列：", slice_2)

# print("原始张量：\n", tensor)

# 访问第 1 个图像的第 2 行第 3 列的元素
# element = tensor[0, 1, 2]  # 等同于 tensor[0][1][2]

# print("访问第 1 个图像的第 2 行第 3 列的元素：", element)
# print(f"tensor[0][1][2] = {tensor[0][1][2]}")


# 获取第 1 个图像的所有行和前 2 列
# slice_1 = tensor[0, :, :2]
# print("获取第 1 个图像的所有行和前 2 列：\n", slice_1)
