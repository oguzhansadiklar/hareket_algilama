import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import time
import threading

class MotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Hareket Algılama Uygulaması')

        self.video_path = ''
        self.cap = None
        self.previous_frame = None
        self.motion_threshold = 50
        self.motion_detected_frames = []

        self.output_directory = 'yakalanan_hareketler'
        os.makedirs(self.output_directory, exist_ok=True)

        self.label = tk.Label(root)
        self.label.pack(pady=10)

        select_video_button = tk.Button(root, text='Video Seç', command=self.select_video)
        select_video_button.pack(pady=5)

        detect_motion_button = tk.Button(root, text='Analiz Et', command=self.detect_motion)
        detect_motion_button.pack(pady=5)

        self.progress_bar = ttk.Progressbar(root, mode='indeterminate')
        self.progress_bar.pack(pady=5)

        self.video_name_label = tk.Label(root, text='', font=('Helvetica', 12))
        self.video_name_label.pack(pady=5)

        # Additional labels for displaying information
        self.process_info_label = tk.Label(root, text='', font=('Helvetica', 12))
        self.process_info_label.pack(pady=5)

        # Label for Oğuzhan Sadıklar's signature
        self.signature_label = tk.Label(root, text='Oğuzhan Sadıklar tarafından yapılmıştır', font=('Helvetica', 8), anchor='sw', justify='left', fg='gray')
        self.signature_label.pack(side='left', anchor='sw', padx=10, pady=10)

        # Variable to keep track of processed frames
        self.processed_frames = 0

        # Thread variable to hold the processing thread
        self.processing_thread = None

    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(self.video_path)
            self.previous_frame = None
            self.motion_detected_frames = []

            # Update video name label
            self.video_name_label.config(text=f'Video Adı: {os.path.basename(self.video_path)}')

            self.show_frame()

    def detect_motion(self):
        # Reset processed frames count
        self.processed_frames = 0

        # Show progress bar
        self.progress_bar.start()

        # Call the function to detect motion in a separate thread
        self.processing_thread = threading.Thread(target=self.detect_motion_frames)
        self.processing_thread.start()

        # Periodically check the thread's status and update the progress info
        self.root.after(100, self.check_thread_status)

    def detect_motion_frames(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.previous_frame is not None:
                diff = cv2.absdiff(self.previous_frame, gray_frame)
                _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

                contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                motion_detected = False
                for contour in contours:
                    if cv2.contourArea(contour) > 500:
                        motion_detected = True
                        break

                if motion_detected:
                    self.motion_detected_frames.append(frame.copy())

                # Update processed frames count
                self.processed_frames += 1

                # Update process info label
                self.update_process_info()

            self.previous_frame = gray_frame

        self.cap.release()

        # Save detected frames
        for i, frame in enumerate(self.motion_detected_frames):
            output_path = os.path.join(self.output_directory, f'{os.path.basename(self.video_path)}_frame_{i}.jpg')
            cv2.imwrite(output_path, frame)

    def check_thread_status(self):
        # Check if the processing thread is still alive
        if self.processing_thread and self.processing_thread.is_alive():
            # Call this method again after 100 milliseconds
            self.root.after(100, self.check_thread_status)
        else:
            # Stop and hide progress bar when the process is finished
            self.progress_bar.stop()
            self.progress_bar.pack_forget()

            # Display process information
            self.process_info_label.config(text=f'{len(self.motion_detected_frames)} tane hareket algılandı')

            # Show a messagebox when the process is finished
            messagebox.showinfo("Hareket Algılama Uygulaması", "Analiz bitti algılanan hareketler için, Tamam'a tıklayınız")

            # Open the folder containing the detected frames
            self.open_output_directory()

    def update_process_info(self):
        # Update process info label
        self.process_info_label.config(text=f'Analiz edlien kare sayısı: {self.processed_frames} ')

    def show_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                height, width, channel = frame.shape
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = img.resize((240, 180), Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(img)
                self.label.config(image=photo)
                self.label.image = photo
            else:
                self.cap.release()
        else:
            self.cap = None

    def open_output_directory(self):
        os.system(f'explorer {os.path.abspath(self.output_directory)}')

    def get_total_frames(self):
        if self.cap is not None:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

if __name__ == "__main__":
    root = tk.Tk()
    app = MotionDetectionApp(root)
    root.geometry("900x700")
    root.mainloop()

