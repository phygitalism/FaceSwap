import os
import cv2
import logging
import argparse

from face_detection import select_face
from face_swap import face_swap

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(lineno)d:%(message)s")

def is_exit_press(timeout=1):
    return cv2.waitKey(timeout) & 0xFF == ord('q')

def is_file(arg):
    if not os.path.isfile(arg):
        raise argparse.ArgumentTypeError("An input is not a file")

    return arg

class VideoHandler(object):
    def __init__(self, src_video: str, dest_video: str, args):
        self._logger = logging.getLogger()
        self.args = args
        self.src_video = cv2.VideoCapture(
            src_video, apiPreference=cv2.CAP_FFMPEG)
        self.dest_video = cv2.VideoCapture(
            dest_video,  apiPreference=cv2.CAP_FFMPEG)
        self.writer = cv2.VideoWriter(args.save_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                      self.dest_video.get(cv2.CAP_PROP_FPS),
                                      (int(self.dest_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                       int(self.dest_video.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    def start(self):
        self._logger.info("Press q to exit")
        frame_num = 1

        window_name = "image"

        if self.args.show:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while self.src_video.isOpened() and self.dest_video.isOpened():
                if is_exit_press():
                    break

                ret_src, src_img = self.src_video.read()
                ret_dst, dst_img = self.dest_video.read()

                if not (ret_src and ret_dst):
                    self._logger.warning("Cannot retrieve frame. Exit")
                    break

                try:
                    src_points, src_shape, src_face = select_face(src_img)

                    if src_points is None:
                        self._logger.warning(
                            "Cannot find facelandmarks on source video. Skip frame: %d", frame_num)
                    else:
                        dst_points, dst_shape, dst_face = select_face(
                            dst_img, choose=False)

                        if dst_points is None:
                            self._logger.warning(
                                "Cannot find facelandmarks on destination video. Skip frame: %d", frame_num)
                        else:
                            dst_img = face_swap(src_face, dst_face, src_points,
                                                dst_points, dst_shape, dst_img, self.args, 68)

                            self.writer.write(dst_img)

                            if self.args.show:
                                cv2.imshow(window_name, dst_img)
                except Exception:
                    self._logger.exception("Unexpected exception. Skip frame")

                frame_num += 1
                self._logger.info("Frame processed: %d", frame_num)
        finally:
            self.src_video.release()
            self.dest_video.release()
            self.writer.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceSwap Video')
    parser.add_argument('--src_video', required=True,
                        help='Path for source video', type=is_file)
    parser.add_argument('--dst_video', required=True,
                        help='Path for dst video', type=is_file)
    parser.add_argument('--warp_2d', action='store_true', help='2d or 3d warp')
    parser.add_argument('--correct_color', action='store_true', help='Correct color')
    parser.add_argument('--show',
                        action='store_true', help='Show')
    parser.add_argument('--save_path', required=True,
                        help='Path for storing output video')
    args = parser.parse_args()

    dir_path = os.path.dirname(args.save_path)

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    VideoHandler(args.src_video, args.dst_video, args).start()
