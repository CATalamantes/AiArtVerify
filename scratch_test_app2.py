from streamlit.testing.v1 import AppTest

at = AppTest.from_file("src/app.py")
at.run(timeout=60)

print("=== After initial load ===")
print("Exception:", at.exception)
for exc in at.exception:
    print("EXC:", exc.value)

title_input = at.text_input[0]
title_input.set_value("THIS IS A HUGE Announcement Video")

category_box = at.selectbox[0]
category_box.set_value("Gaming")

tag_slider = at.slider[0]
tag_slider.set_value(25)

hour_slider = at.slider[1]
hour_slider.set_value(18)

day_box = at.selectbox[1]
day_box.set_value("Friday")

at.button[0].click().run(timeout=60)

print()
print("=== After predict click ===")
print("Exception:", at.exception)
for exc in at.exception:
    print("EXC:", exc.value)
    import traceback
    print(exc.stack_trace if hasattr(exc, 'stack_trace') else '')

print("Metrics:", [(m.label, m.value) for m in at.metric])
print("Captions:", [c.value for c in at.caption])
print("Infos:", [i.value for i in at.info])
print("Markdown headers:", [m.value for m in at.markdown if '#' in m.value])

# Inspect the underlying chart element for the contribution data
charts = at.get("arrow_vega_lite_chart")
print("Number of vega-lite charts found:", len(charts))
