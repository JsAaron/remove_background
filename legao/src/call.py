from config import config
from main import legao_main

def prepare():
    legao_main(
        original_dir=config["original_dir"],
        target_dir=config["target_dir"],
        error_dir=config["error_dir"],
        start_y=config["start_y"],
        resize=config["resize"],
        collectData=config["collectData"],
        splitMode=config["splitMode"],
        interval=config["interval"],
        end_y=config["end_y"])

if __name__ == "__main__":
    prepare()
