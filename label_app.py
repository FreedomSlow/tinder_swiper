import os
from PIL import Image, ImageTk
import pickle
import tkinter

from helpers import read_pickle


def on_close():
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(labels_, f)
        window.destroy()


def next_profile():
    global cnt, current_profile, profile_images
    cnt = -1

    try:
        current_profile = next(not_labeled_profiles)
    except StopIteration:
        print("All images are labeled! Closing the app")
        on_close()
    profile_images = list(filter(lambda path: path.startswith(current_profile), all_images_))
    next_image()
    display_bio(current_profile)


def display_image(path: str):
    img = Image.open(path)
    width, height = img.size
    max_height = 800
    if height > max_height:
        resize_factor = max_height / height
        img = img.resize((int(width * resize_factor), int(height * resize_factor)), resample=Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    img_lbl.img = img_tk
    img_lbl.config(image=img_lbl.img)


def make_pretty_text(text, max_len=90):
    return "\n".join([text[part * max_len: (part + 1) * max_len] for part in range(len(text) // max_len)])


def display_bio(profile_id: str):
    try:
        bio_text.config(text=make_pretty_text([profile_id]["bio"]))
        name_age.config(
            text=f"{bios[profile_id]['name']}: {bios[profile_id]['age']}",
            font=("Helvetica", 32)
        )
    except KeyError:
        bio_text.config(text="No bio available")


def next_image():
    global cnt, profile_images
    if cnt < len(profile_images):
        cnt = min(len(profile_images) - 1, cnt + 1)
        display_image(os.path.join(IMAGES_DIR, profile_images[cnt]))
    else:
        pass


def prev_image():
    global cnt, profile_images
    if cnt >= 0:
        cnt = max(0, cnt - 1)
        display_image(os.path.join(IMAGES_DIR, profile_images[cnt]))
    else:
        pass


def positive():
    global current_profile
    labels_[current_profile] = 1
    next_profile()


def negative():
    global current_profile
    labels_[current_profile] = 0
    next_profile()


if __name__ == "__main__":
    LABELS_PATH = "data/labels.pkl"
    IMAGES_DIR = "data/images"
    BIOS_DIR = "data/bios.pkl"

    labels_ = read_pickle(LABELS_PATH, {})
    all_images_ = [f for f in os.listdir(IMAGES_DIR) if os.path.isfile(os.path.join(IMAGES_DIR, f))]
    all_profiles_ = set([f.split("_")[0] for f in all_images_])
    not_labeled_profiles = iter(all_profiles_ - set(labels_.keys()))

    bios = read_pickle(BIOS_DIR, {})

    print(f"Total profiles: {len(all_profiles_)}")
    print(f"Not labeled profiles: {len(all_profiles_ - set(labels_.keys()))}")

    window = tkinter.Tk()
    window.geometry("720x1080")

    img_lbl = tkinter.Label(window)
    img_lbl.pack(anchor="center")

    name_age = tkinter.Label(window)
    name_age.place(relx=0.5, rely=0.8, anchor="center")

    bio_text = tkinter.Label(window)
    bio_text.place(relx=0.5, rely=0.85, anchor="center")

    btn_like = tkinter.Button(window, text="Like", fg="green", command=positive)
    btn_like.place(relx=0.7, rely=0.9, anchor="center")

    btn_dislike = tkinter.Button(window, text="Dislike", fg="red", command=negative)
    btn_dislike.place(relx=0.3, rely=0.9, anchor="center")

    btn_prev = tkinter.Button(window, text="Prev Image", command=prev_image)
    btn_prev.place(relx=0.3, rely=0.95, anchor="center")

    btn_next = tkinter.Button(window, text="Next Image", command=next_image)
    btn_next.place(relx=0.7, rely=0.95, anchor="center")

    next_profile()

    window.protocol("WM_DELETE_WINDOW", on_close)
    window.mainloop()
