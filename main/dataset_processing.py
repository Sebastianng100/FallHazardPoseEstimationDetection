import os
import shutil

KAGGLE_DIR = "../kaggle_fall_dataset"
ANNOTATED_DIR = "../annotated_fall_dataset"

OUTPUT_DIR = "../processed_dataset"

SUBSETS = ["train", "val"]
possible_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

def process_kaggle():
    for subset in SUBSETS:
        img_in = os.path.join(KAGGLE_DIR, "images", subset)
        lbl_in = os.path.join(KAGGLE_DIR, "labels", subset)

        img_out = os.path.join(OUTPUT_DIR, "images", subset)
        lbl_out = os.path.join(OUTPUT_DIR, "labels", subset)

        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for lbl_file in os.listdir(lbl_in):
            if not lbl_file.endswith(".txt"):
                continue

            base_name = os.path.splitext(lbl_file)[0]

            src_img = None
            for ext in possible_exts:
                candidate = os.path.join(img_in, base_name + ext)
                if os.path.exists(candidate):
                    src_img = candidate
                    break

            if src_img is None:
                print(f"No image found for {lbl_file}")
                continue

            src_lbl = os.path.join(lbl_in, lbl_file)
            dst_img = os.path.join(img_out, os.path.basename(src_img))
            dst_lbl = os.path.join(lbl_out, lbl_file)

            shutil.copy(src_img, dst_img)

            with open(src_lbl, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    if cls == 0:
                        new_cls = 0
                    else:
                        new_cls = 1
                    parts[0] = str(new_cls)
                    new_lines.append(" ".join(parts))

            with open(dst_lbl, "w") as f:
                f.write("\n".join(new_lines))


def process_annotated():
    img_in = os.path.join(ANNOTATED_DIR, "images")
    lbl_in = os.path.join(ANNOTATED_DIR, "labels")

    img_out = os.path.join(OUTPUT_DIR, "images", "train")
    lbl_out = os.path.join(OUTPUT_DIR, "labels", "train")

    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    for lbl_file in os.listdir(lbl_in):
        if not lbl_file.endswith(".txt"):
            continue

        base_name = os.path.splitext(lbl_file)[0]

        src_img = None
        for ext in possible_exts:
            candidate = os.path.join(img_in, base_name + ext)
            if os.path.exists(candidate):
                src_img = candidate
                break

        if src_img is None:
            print(f"No image found for {lbl_file}")
            continue

        src_lbl = os.path.join(lbl_in, lbl_file)

        dst_img = os.path.join(img_out, os.path.basename(src_img))
        dst_lbl = os.path.join(lbl_out, lbl_file)

        shutil.copy(src_img, dst_img)
        shutil.copy(src_lbl, dst_lbl)


if __name__ == "__main__":
    process_kaggle()
    process_annotated()
    print("Processing has been successfully completed.")