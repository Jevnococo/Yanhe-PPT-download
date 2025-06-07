import os
from PIL import Image, ImageDraw # 添加 ImageDraw
from reportlab.lib.pagesizes import letter, A4 # 可以选择纸张大小
from reportlab.pdfgen import canvas

def images_to_pdf(image_paths, output_pdf_path, page_size=A4):
    """
    将一组图片拼接成一个 PDF 文件。
    每张图片占据 PDF 的一页，并尽可能大地适应页面（保持宽高比）。

    参数:
    image_paths (list): 包含图片文件路径的列表。图片会按照列表中的顺序添加到 PDF 中。
    output_pdf_path (str): 输出 PDF 文件的路径。
    page_size (tuple): PDF 页面的尺寸，例如 letter, A4。默认为 A4。
                       可以自定义元组 (width, height)，单位为 points (1 inch = 72 points)。
    """
    if not image_paths:
        print("错误：图片路径列表为空。")
        return

    # 检查输出路径的目录是否存在，如果不存在则创建
    output_dir = os.path.dirname(output_pdf_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"创建目录：{output_dir}")
        except OSError as e:
            print(f"错误：无法创建目录 {output_dir}: {e}")
            return

    # 创建 PDF 画布
    c = canvas.Canvas(output_pdf_path, pagesize=page_size)
    page_width, page_height = page_size  # 获取页面尺寸

    print(f"开始生成 PDF: {output_pdf_path}")
    print(f"页面尺寸: {page_width} x {page_height} points")

    for i, image_path in enumerate(image_paths):
        try:
            print(f"处理图片 ({i+1}/{len(image_paths)}): {image_path}")
            img = Image.open(image_path)
            img_width, img_height = img.size

            # 计算缩放比例以适应页面，同时保持宽高比
            aspect_ratio = img_width / float(img_height)
            available_width = page_width * 0.9  # 留一些边距
            available_height = page_height * 0.9 # 留一些边距

            # 根据宽高比确定最终绘制的图片尺寸
            if (available_width / aspect_ratio) <= available_height:
                draw_width = available_width
                draw_height = available_width / aspect_ratio
            else:
                draw_height = available_height
                draw_width = available_height * aspect_ratio

            # 计算图片在页面上的位置（居中）
            x_offset = (page_width - draw_width) / 2
            y_offset = (page_height - draw_height) / 2

            # 在 PDF 上绘制图片
            # reportlab 的 drawImage 使用的是图片的左下角作为原点
            # ImageReader 是 reportlab 处理 PIL Image 对象的方式
            c.drawImage(image_path, x_offset, y_offset, width=draw_width, height=draw_height, preserveAspectRatio=True, anchor='c')
            c.showPage()  # 创建新的一页

        except FileNotFoundError:
            print(f"错误：图片文件未找到: {image_path}")
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")

    try:
        c.save()
        print(f"PDF 文件已成功保存到: {output_pdf_path}")
    except Exception as e:
        print(f"保存 PDF 文件时发生错误: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 假设你的图片在 'my_images' 文件夹下
    image_folder = r"D:\Yanhe\extracted_ppt_slides_test\slides_14_5_2_5"
    output_pdf_name = r"D:\Yanhe\extracted_ppt_slides_test\slides_14_5_2_5" + r"\combined_slides.pdf"

    # 确保图片文件夹存在
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        print(f"创建了示例图片文件夹: {image_folder}")
        print("请在该文件夹中放入一些图片 (例如 .jpg, .png)。")
        # 你可以手动创建一些示例图片，或者跳过生成
        # 例如，创建一些简单的占位符图片 (需要Pillow)
        try:
            for i in range(3):
                img = Image.new('RGB', (600, 800), color = ('red' if i % 2 == 0 else 'blue'))
                d = ImageDraw.Draw(img)
                d.text((10,10), f"Sample Image {i+1}", fill=(255,255,0))
                img.save(os.path.join(image_folder, f"sample_image_{i+1}.png"))
            print("已生成3张示例图片。")
        except NameError: # ImageDraw 未导入
            pass
        except Exception as e:
            print(f"创建示例图片时出错: {e}")


    # 获取文件夹中所有的图片文件 (支持常见格式)
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
    images = []
    if os.path.exists(image_folder):
        for filename in sorted(os.listdir(image_folder)): # sorted() 确保按文件名顺序
            if filename.lower().endswith(supported_formats):
                images.append(os.path.join(image_folder, filename))
    else:
        print(f"错误：图片文件夹 '{image_folder}' 不存在。")


    if images:
        # 指定输出 PDF 文件的完整路径
        # 例如，保存在脚本同级目录下
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_output_path = os.path.join(script_dir, output_pdf_name)

        images_to_pdf(images, pdf_output_path, page_size=A4) # 你也可以使用 letter 或自定义尺寸

        # 如果你想使用其他页面尺寸，例如 letter：
        # images_to_pdf(images, "combined_letter.pdf", page_size=letter)

        # 如果你想自定义页面尺寸 (单位是 points, 1 inch = 72 points)
        # 例如，一个正方形页面
        # custom_size = (500, 500) # 500x500 points
        # images_to_pdf(images, "combined_custom.pdf", page_size=custom_size)
    else:
        print(f"在文件夹 '{image_folder}' 中没有找到支持的图片文件。")