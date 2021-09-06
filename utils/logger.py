import sys
import logging

def init_logging(logging_path):
    file_writer = logging.FileHandler(logging_path)
    formatter = logging.Formatter('%(asctime)s %(name)-3s %(levelname)-3s %(message)s')
    file_writer.setLevel(logging.INFO)
    file_writer.setFormatter(formatter)

    console_writer = logging.StreamHandler(sys.stdout)
    console_writer.setLevel(logging.INFO)

    root = logging.getLogger()
    root.addHandler(file_writer)
    root.addHandler(console_writer)
    root.setLevel(logging.INFO)


