import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import uuid
import time # For profiling if needed
from concurrent.futures import ProcessPoolExecutor, as_completed # For multiprocessing



# 缩放图像进行SSIM计算的尺寸
SSIM_RESIZE_WIDTH = 640 
SSIM_RESIZE_HEIGHT = 360 

def resize_for_ssim(frame_gray):
    """将灰度帧缩放到固定大小以便更快地计算SSIM"""
    return cv2.resize(frame_gray, (SSIM_RESIZE_WIDTH, SSIM_RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)

def extract_ppt_slides(video_path, output_dir="ppt_slides_output", ssim_threshold=0.95, frame_skip=30, min_slide_duration_frames=3):
    """
    从视频中提取PPT幻灯片。
    - ssim_threshold: 比较帧相似度的阈值。
    - frame_skip: 每隔多少帧处理一次。
    - min_slide_duration_frames: 新幻灯片需要稳定多少个“已处理帧”才保存。
    """
    # 确保使用绝对路径
    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(video_path):
        print(f"错误: 视频文件 '{video_path}' 未找到。")
        return 0

    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"创建目录 '{output_dir}' 出错: {e}")
        return 0

    test_file = os.path.join(output_dir, "test_write.txt")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print(f"目录 '{output_dir}' 写入测试失败: {e}. 请检查权限。")
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 '{video_path}'。")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps == 0: # Handle cases where FPS might not be readable
        print(f"警告: 无法获取视频FPS，将使用默认值30进行估算。总帧数={total_frames}")
        fps = 30.0
    else:
        print(f"视频信息: FPS={fps:.2f}, 总帧数={total_frames}")

    actual_min_duration_seconds = (frame_skip * min_slide_duration_frames) / fps
    print(f"配置: SSIM阈值={ssim_threshold}, 帧跳过={frame_skip}, "
          f"最小幻灯片稳定处理帧数={min_slide_duration_frames} "
          f"(约 {actual_min_duration_seconds:.2f} 秒真实视频时长)")

    saved_slide_gray_ssim = None # 用于SSIM比较的缩放灰度图
    slide_count = 0
    frame_number = 0
    processed_frame_count = 0

    potential_new_slide_frame_color = None # 保存原始彩色帧
    potential_new_slide_gray_ssim = None # 保存用于比较的缩放灰度帧
    potential_new_slide_consecutive_checks = 0

    print(f"开始处理视频: {os.path.basename(video_path)}...")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break # 视频结束或读取错误

        frame_number += 1

        if frame_number % frame_skip == 0:
            processed_frame_count += 1
            current_frame_gray_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_frame_gray_ssim = resize_for_ssim(current_frame_gray_orig)

            if saved_slide_gray_ssim is None: # 第一张幻灯片
                print(f"帧 {frame_number}: 检测到初始幻灯片。")
                slide_filename = os.path.join(output_dir, f"slide_{slide_count:04d}_{frame_number}.png")
                try:
                    if cv2.imwrite(slide_filename, frame):
                        print(f"  已保存: {slide_filename}")
                        saved_slide_gray_ssim = current_frame_gray_ssim # 保存缩放后的灰度图
                        slide_count += 1
                    else:
                        print(f"  错误: 无法保存图片到 '{slide_filename}'")
                except Exception as e:
                    print(f"  保存图片时出现异常: {e}")
                potential_new_slide_consecutive_checks = 0
                potential_new_slide_frame_color = None
            else:
                try:
                    # 确保形状一致 (resize_for_ssim 已经保证了)
                    s = ssim(saved_slide_gray_ssim, current_frame_gray_ssim)
                except ValueError as e: # 比如窗口大小问题
                    print(f"  帧 {frame_number}: SSIM计算错误 - {e}. 跳过比较。")
                    s = 1.0 # 假设无变化，避免错误中断

                if s < ssim_threshold: # 检测到与上一张已保存幻灯片不同
                    if potential_new_slide_frame_color is None:
                        # 这是第一次检测到这个潜在的新幻灯片
                        # print(f"帧 {frame_number}: 发现潜在新幻灯片 (与已存幻灯片SSIM={s:.4f})。开始观察稳定性...")
                        potential_new_slide_frame_color = frame # 保存原始彩色帧
                        potential_new_slide_gray_ssim = current_frame_gray_ssim
                        potential_new_slide_consecutive_checks = 1
                    else:
                        # 已经有一个潜在幻灯片了，比较当前帧和这个“潜在”幻灯片是否相似
                        try:
                            s_potential = ssim(potential_new_slide_gray_ssim, current_frame_gray_ssim)
                        except ValueError as e:
                            print(f"  帧 {frame_number}: 潜在幻灯片SSIM计算错误 - {e}. 重置观察。")
                            s_potential = 0.0 # 认为非常不同，重置

                        if s_potential > ssim_threshold: # 当前帧与“潜在”幻灯片相似，继续观察
                            potential_new_slide_consecutive_checks += 1
                            # 更新为当前帧，以应对非常缓慢的过渡效果，并取最新的稳定帧
                            potential_new_slide_frame_color = frame
                            potential_new_slide_gray_ssim = current_frame_gray_ssim
                        else: # 当前帧与“潜在”幻灯片不同，说明之前的“潜在”不稳定，用当前帧作为新的“潜在”
                            # print(f"帧 {frame_number}: 潜在幻灯片变化 (与上一潜在SSIM={s_potential:.4f})。重置观察...")
                            potential_new_slide_frame_color = frame
                            potential_new_slide_gray_ssim = current_frame_gray_ssim
                            potential_new_slide_consecutive_checks = 1
                    
                    if potential_new_slide_consecutive_checks >= min_slide_duration_frames:
                        print(f"帧 {frame_number}: 确认新幻灯片 (稳定观察 {potential_new_slide_consecutive_checks} 次)。")
                        slide_filename = os.path.join(output_dir, f"slide_{slide_count:04d}_{frame_number}.png")
                        try:
                            if cv2.imwrite(slide_filename, potential_new_slide_frame_color): # 保存原始彩色帧
                                print(f"  已保存: {slide_filename}")
                                saved_slide_gray_ssim = potential_new_slide_gray_ssim # 更新已保存的幻灯片（的SSIM比较版本）
                                slide_count += 1
                            else:
                                print(f"  错误: 无法保存图片到 '{slide_filename}'")
                        except Exception as e:
                            print(f"  保存图片时出现异常: {e}")
                        
                        potential_new_slide_frame_color = None
                        potential_new_slide_gray_ssim = None
                        potential_new_slide_consecutive_checks = 0
                else: # 与上一张已保存幻灯片相似，重置潜在幻灯片观察
                    if potential_new_slide_frame_color is not None:
                        # print(f"帧 {frame_number}: 之前潜在的幻灯片未稳定，恢复到已保存幻灯片状态 (SSIM={s:.4f})。")
                        pass
                    potential_new_slide_frame_color = None
                    potential_new_slide_gray_ssim = None
                    potential_new_slide_consecutive_checks = 0

            # 移除 previous_frame_gray，因为它在当前逻辑下没有被直接用来与当前帧比较，而是与 saved_slide_gray 比较

        # 打印进度
        if frame_number % (frame_skip * 20) == 0: # 每处理 frame_skip * 20 个原始帧打印一次
             progress = (frame_number / total_frames) * 100 if total_frames > 0 else 0
             elapsed_time = time.time() - start_time
             print(f"  已处理帧: {frame_number}/{total_frames} ({progress:.2f}%) "
                   f"耗时: {elapsed_time:.2f}s. 已提取: {slide_count}张.")

    cap.release()
    end_time = time.time()
    print(f"\n视频 '{os.path.basename(video_path)}' 处理完成! 总共提取了 {slide_count} 张幻灯片到 '{output_dir}'.")
    print(f"总耗时: {end_time - start_time:.2f} 秒.")
    return slide_count

def process_single_video_wrapper(args):
    """帮助函数，用于multiprocessing.Pool.map"""
    video_path, output_dir, ssim_thr, fs, min_dur = args
    return extract_ppt_slides(video_path, output_dir, ssim_thr, fs, min_dur)

def process_videos_in_folder(input_folder, output_base_folder="extracted_slides", 
                             ssim_threshold=0.97, frame_skip=30, min_slide_duration_frames=3,
                             max_workers=None):
    """
    处理指定文件夹中的所有视频文件。
    max_workers: 并行处理视频的最大进程数，None 表示使用 os.cpu_count()。
    """
    input_folder = os.path.abspath(input_folder)
    output_base_folder = os.path.abspath(output_base_folder)

    if not os.path.exists(input_folder):
        print(f"错误: 输入文件夹 '{input_folder}' 不存在。")
        return

    try:
        os.makedirs(output_base_folder, exist_ok=True)
    except Exception as e:
        print(f"创建输出基础目录 '{output_base_folder}' 时出错: {e}")
        return

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    try:
        all_files = os.listdir(input_folder)
        video_files = [f for f in all_files if os.path.splitext(f)[1].lower() in video_extensions]
    except Exception as e:
        print(f"读取输入目录 '{input_folder}' 时出错: {e}")
        return

    if not video_files:
        print(f"警告: 在文件夹 '{input_folder}' 中未找到视频文件。")
        return

    print(f"在文件夹 '{input_folder}' 中找到 {len(video_files)} 个视频文件。")
    print(f"全局配置: SSIM阈值={ssim_threshold}, 帧跳过={frame_skip}, 最小幻灯片稳定处理帧数={min_slide_duration_frames}")

    tasks = []
    for i, video_file in enumerate(video_files):
        video_path = os.path.join(input_folder, video_file)
        video_name_safe = "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in os.path.splitext(video_file)[0])
        video_name_safe = video_name_safe.replace(' ', '_')[:50] # 限制长度并替换空格
        
        video_output_dir = os.path.join(output_base_folder, f"slides_{video_name_safe}_{i+1}")
        
        tasks.append((video_path, video_output_dir, ssim_threshold, frame_skip, min_slide_duration_frames))

    total_slides_extracted_all_videos = 0
    processed_video_count = 0
    
    # 使用 ProcessPoolExecutor 进行并行处理
    # 如果 max_workers 为 None，它将默认为机器的CPU核心数
    # 对于I/O密集型或混合型任务，可以尝试核心数+1
    if max_workers is None:
        max_workers = os.cpu_count()
    print(f"将使用最多 {max_workers} 个进程并行处理视频...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {executor.submit(process_single_video_wrapper, task): task[0] for task in tasks}
        
        for i, future in enumerate(as_completed(future_to_video)):
            video_path_processed = future_to_video[future]
            try:
                slides_extracted_count = future.result()
                if slides_extracted_count > 0:
                    print(f"视频 '{os.path.basename(video_path_processed)}' 处理成功，提取了 {slides_extracted_count} 张幻灯片。")
                    total_slides_extracted_all_videos += slides_extracted_count
                    processed_video_count +=1
                elif slides_extracted_count == 0:
                     print(f"视频 '{os.path.basename(video_path_processed)}' 处理完成，但未提取到幻灯片。")
                else: # 可能是负数或None，表示处理中出现错误但被捕获返回了
                    print(f"视频 '{os.path.basename(video_path_processed)}' 处理时可能遇到问题，返回值: {slides_extracted_count}")

            except Exception as exc:
                print(f"视频 '{os.path.basename(video_path_processed)}' 在并行处理中产生异常: {exc}")
            
            print(f"--- 总体进度: {i+1}/{len(tasks)} 个视频任务完成 ---")


    print(f"\n所有视频处理完成!")
    print(f"成功处理了 {processed_video_count}/{len(video_files)} 个视频。")
    print(f"总共提取了 {total_slides_extracted_all_videos} 张幻灯片。")
    print(f"幻灯片保存在 '{output_base_folder}' 目录下的各个子文件夹中。")

# --- 使用示例 ---
if __name__ == "__main__":
    input_folder_path = r"D:\Yanhe\release_downloader_2\output\智能感知与信息处理-screen"  # 视频文件夹路径
    

    output_base_folder_path = r"D:\Yanhe\extracted_ppt_slides_test"


   
    
    print("开始批量处理视频...")
    process_videos_in_folder(
        input_folder_path,
        output_base_folder=output_base_folder_path,
        ssim_threshold=0.97,         
        frame_skip=250,                
        min_slide_duration_frames=2,  # 新幻灯片稳定出现2个已处理帧
        max_workers=max(1, os.cpu_count() - 1) if os.cpu_count() else 1 # 使用 CPU核心数-1 个进程,至少1个
    )