"""
nii文件生成stl代码
用于将nii文件生成stl文件

"""


import vtk
import numpy as np

# 读取.nii.gz文件并转换为numpy数组
reader = vtk.vtkNIFTIImageReader()
reader.SetFileName(r"C:\Users\allstar\Desktop\aaaaa\liver-213\ct_213_0000.nii.gz")
reader.Update()
imageData = reader.GetOutput()

# 提取Marching Cubes等高线
contour = vtk.vtkMarchingCubes()
contour.SetInputData(imageData)
contour.SetValue(0, 1)
contour.Update()

# 应用Laplacian平滑
smoothFilter = vtk.vtkSmoothPolyDataFilter()
smoothFilter.SetInputConnection(contour.GetOutputPort())
smoothFilter.SetNumberOfIterations(50)  # 迭代次数，可以根据需要调整
smoothFilter.SetRelaxationFactor(0.1)  # 松弛因子，可以根据需要调整
smoothFilter.Update()

# 保存平滑后的模型
writer = vtk.vtkSTLWriter()
writer.SetFileName(r"C:\Users\allstar\Desktop\aaaaa\liver.stl")
writer.SetInputConnection(smoothFilter.GetOutputPort())
writer.Write()
print("completed")