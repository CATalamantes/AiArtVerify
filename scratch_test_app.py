from streamlit.testing.v1 import AppTest

at = AppTest.from_file("src/app.py")
at.run()

print("=== After initial load ===")
print("Exception:", at.exception)
print("Title text:", [t.value for t in at.title])
print("Text inputs:", [(w.label, w.value) for w in at.text_input])
print("Selectboxes:", [(w.label, w.value) for w in at.selectbox])
print("Number inputs:", [(w.label, w.value) for w in at.number_input])
print("Sliders:", [(w.label, w.value) for w in at.slider])

# Fill the form
at.text_input(key=None).set_value if False else None  # noop, placeholder
title_input = at.text_input[0]
title_input.set_value("THIS IS A HUGE Announcement Video")

category_box = at.selectbox[0]
category_box.set_value("Gaming")

tag_count_input = at.number_input[0]
tag_count_input.set_value(25)

hour_slider = at.slider[0]
hour_slider.set_value(18)

day_box = at.selectbox[1]
day_box.set_value("Friday")

at.button[0].click().run()

print()
print("=== After form submit ===")
print("Exception:", at.exception)
if at.exception:
    for exc in at.exception:
        print("EXC:", exc.value)
print("Metrics:", [(m.label, m.value) for m in at.metric])
print("Captions:", [c.value for c in at.caption])
