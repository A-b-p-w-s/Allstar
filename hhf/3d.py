# import vtk
# import numpy as np
# import nibabel as nib



# nii_image1 = nib.load(r'C:\Users\allstar\Desktop\body_model\new_tumour\tumour.nii.gz')  # 替换为你的文件路径
# image_data = nii_image1.get_fdata()
# data = np.array(image_data)
# x = nii_image1.header['pixdim']


# nii_image2 = nib.load(r'C:\Users\allstar\Desktop\body_model\bone\bone.nii')  # 替换为你的文件路径
# bone_data = nii_image2.get_fdata()
# # bone_data[bone_data>0]=2
# bone_data = np.array(bone_data)

# nii_image3 = nib.load(r'C:\Users\allstar\Desktop\body_model\body_model\Segmentation.nii')  # 替换为你的文件路径
# body_data = nii_image3.get_fdata()

# nii_image4 = nib.load(r'C:\Users\allstar\Desktop\body_model\Marker\processed_image_abs.nii.gz')  # 替换为你的文件路径
# marker = nii_image4.get_fdata()

# body_data[body_data>0]=10
# body_data = np.array(body_data)
# data = data + bone_data + body_data + marker

# # 将NumPy数组转换为VTK格式
# vtk_data = vtk.vtkImageData()
# vtk_data.SetDimensions(data.shape[2], data.shape[1], data.shape[0])
# vtk_data.SetSpacing(x[3], x[2], x[1])  # 根据需要调整间距
# vtk_data.AllocateScalars(vtk.VTK_FLOAT, 1)
# # 将数据填充到VTK图像数据中
# for z in range(data.shape[0]):
#     for y in range(data.shape[1]):
#         for x in range(data.shape[2]):
#             vtk_data.SetScalarComponentFromDouble(x, y, z, 0, data[z, y, x])
            

# # 创建体渲染器
# volume_ray_cast_mapper = vtk.vtkSmartVolumeMapper()
# volume_ray_cast_mapper.SetInputData(vtk_data)

# # 创建体属性
# volume_property = vtk.vtkVolumeProperty()
# volume_property.ShadeOn()
# volume_property.SetInterpolationTypeToLinear()

# # 设置透明度函数
# opacity_function = vtk.vtkPiecewiseFunction()
# opacity_function.AddPoint(0, 0.0)  # 值为0时完全透明
# opacity_function.AddPoint(10, 0.002)  # 可以根据需要调整透明度
# opacity_function.AddPoint(11, 1)  # 可以根据需要调整透明度
# opacity_function.AddPoint(12, 1)  # 可以根据需要调整透明度
# opacity_function.AddPoint(13, 1)  # 可以根据需要调整透明度

# volume_property.SetScalarOpacity(opacity_function)

# # 设置颜色传输函数
# color_function = vtk.vtkColorTransferFunction()
# color_function.AddRGBPoint(0, 0.0, 0.0, 0.0)
# color_function.AddRGBPoint(1, 1.0, 1.0, 1.0)
# volume_property.SetColor(color_function)

# # 创建体对象
# volume = vtk.vtkVolume()
# volume.SetMapper(volume_ray_cast_mapper)
# volume.SetProperty(volume_property)

# # 创建渲染器、渲染窗口和交互器
# renderer = vtk.vtkRenderer()
# renderer.AddVolume(volume)
# renderer.SetBackground(0.1, 0.2, 0.3)  # 设置背景颜色

# render_window = vtk.vtkRenderWindow()
# render_window.AddRenderer(renderer)
# render_window.SetSize(300, 300)

# render_window_interactor = vtk.vtkRenderWindowInteractor()
# render_window_interactor.SetRenderWindow(render_window)

# # 初始化交互器并开始渲染
# render_window_interactor.Initialize()
# render_window.Render()
# render_window_interactor.Start()



import vtk
import numpy as np
import nibabel as nib

nii_image1 = nib.load(r'C:\Users\allstar\Desktop\body_model\new_tumour\tumour.nii.gz')  # 替换为你的文件路径
image_data = nii_image1.get_fdata()
data = np.array(image_data)
pixdim = nii_image1.header['pixdim']

nii_image2 = nib.load(r'C:\Users\allstar\Desktop\body_model\bone\bone.nii')  # 替换为你的文件路径
bone_data = nii_image2.get_fdata()
bone_data = np.array(bone_data)

nii_image3 = nib.load(r'C:\Users\allstar\Desktop\body_model\body_model\Segmentation.nii')  # 替换为你的文件路径
body_data = nii_image3.get_fdata()

nii_image4 = nib.load(r'C:\Users\allstar\Desktop\body_model\Marker\processed_image_abs.nii.gz')  # 替换为你的文件路径
marker = nii_image4.get_fdata()

body_data[body_data>0]=10
body_data = np.array(body_data)
data = data + bone_data + body_data + marker

# 将NumPy数组转换为VTK格式
vtk_data = vtk.vtkImageData()
vtk_data.SetDimensions(data.shape[2], data.shape[1], data.shape[0])
vtk_data.SetSpacing(pixdim[3], pixdim[2], pixdim[1])  # 根据需要调整间距
vtk_data.AllocateScalars(vtk.VTK_FLOAT, 1)
# 将数据填充到VTK图像数据中
for z in range(data.shape[0]):
    for y in range(data.shape[1]):
        for x in range(data.shape[2]):
            vtk_data.SetScalarComponentFromDouble(x, y, z, 0, data[z, y, x])
            
# 创建体渲染器
volume_ray_cast_mapper = vtk.vtkSmartVolumeMapper()
volume_ray_cast_mapper.SetInputData(vtk_data)

# 创建体属性
volume_property = vtk.vtkVolumeProperty()
volume_property.ShadeOn()
volume_property.SetInterpolationTypeToLinear()

# 设置透明度函数
opacity_function = vtk.vtkPiecewiseFunction()
opacity_function.AddPoint(0, 0.0)  # 值为0时完全透明
opacity_function.AddPoint(10, 0.002)  # 可以根据需要调整透明度
opacity_function.AddPoint(11, 1)  # 可以根据需要调整透明度
opacity_function.AddPoint(12, 1)  # 可以根据需要调整透明度
opacity_function.AddPoint(13, 1)  # 可以根据需要调整透明度

volume_property.SetScalarOpacity(opacity_function)

# 设置颜色传输函数
color_function = vtk.vtkColorTransferFunction()
color_function.AddRGBPoint(0, 0.0, 0.0, 0.0)
color_function.AddRGBPoint(1, 1.0, 1.0, 1.0)
volume_property.SetColor(color_function)

# 创建体对象
volume = vtk.vtkVolume()
volume.SetMapper(volume_ray_cast_mapper)
volume.SetProperty(volume_property)

# 创建渲染器、渲染窗口和交互器
renderer = vtk.vtkRenderer()
renderer.AddVolume(volume)
renderer.SetBackground(0.1, 0.2, 0.3)  # 设置背景颜色

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(300, 300)

render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# 添加线段
line_source = vtk.vtkLineSource()
line_source.SetPoint1(153*pixdim[3], 103*pixdim[2], 271*pixdim[1])
line_source.SetPoint2(153*pixdim[3], 186*pixdim[2], 271*pixdim[1])
line_source.Update()

line_mapper = vtk.vtkPolyDataMapper()
line_mapper.SetInputConnection(line_source.GetOutputPort())

line_actor = vtk.vtkActor()
line_actor.SetMapper(line_mapper)
line_actor.GetProperty().SetColor(1, 0, 0)  # 设置线段颜色为红色
renderer.AddActor(line_actor)

# 初始化交互器并开始渲染
render_window_interactor.Initialize()
render_window.Render()
render_window_interactor.Start()