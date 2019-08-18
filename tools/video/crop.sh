out_w=800
out_h=600
x=400
y=60
ffmpeg -i input.mp4 -filter:v "crop=${out_w}:${out_h}:${x}:${y}" output.mp4