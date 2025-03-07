
# EOF
# red #FF0000
# orange  #FFA500
# yellow  #FFFF00
# green #00FF00
# cyan  #00FFFF
# blue  #0000FF
# skyblue #87CEEB
# purple  #A020F0
# white #FFFFFF
# pink  #FF69B4
# pinkRed #FFC0CB
# fuchsia #FF00FF
# lightgreen  #90EE90
# brown  #CC7722
# orangered  #FF4500
# gray   #808080
# golden   #FFD700
# babyblue   #E0FFFF
# beige  #F5F5DC
# deepblue   #191970
# coffee   #4D3900
# crimson  #DC143C
# navyblue   #000080
# violet   #7F00FF
# darkgreen  #006400
# mauve  #E0B0FF
# dodgerblue   #1E90FF
# black  #000000
# darkslategray  #2F4F4F
# silver   #C0C0C0
# <<EOF

##colors="babyblue,random,purple,violet,orangered,blue,beige,pinkRed,crimson,lightgreen,silver,orange,darkslategray,green,black,skyblue,gray,coffee,yellow,brown,deepblue,mauve,darkgreen,pink,navyblue,cyan,white,golden,fuchsia,dodgerblue,red"
##colors="浅蓝色,彩光色,紫色,紫罗兰色,橘红色,蓝光色,米色,粉红色,桃红色,浅绿色,银色,橘色,墨绿色,绿色,黑色,天蓝色,灰色,咖啡色,黄光色,棕色,深蓝色,浅紫色,深绿色,粉色,藏青色,青色,白色,金色,紫红色,湖蓝色,红色"

colors="橘色,黄光色,绿色,青色,蓝光色,天蓝色,紫色,白色,粉色,粉红色,紫红色,浅绿色,棕色,橘红色,灰色,金色,浅蓝色,米色,深蓝色,咖啡色,桃红色,藏青色,紫罗兰色,深绿色,浅紫色,湖蓝色,黑色,墨绿色,银色"
#colors="orange,yellow,green,cyan,blue,skyblue,purple,white,pink,pinkRed,fuchsia,lightgreen,brown,orangered,gray,golden,babyblue,beige,deepblue,coffee,crimson,navyblue,violet,darkgreen,mauve,dodgerblue,black,darkslategray,silver"
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/docker_model_input_path",
        "prompt": "请将以下的英文颜色名的转换成对应的16进制的RGB值，如'red'->'#FF0000'，输出格式为:颜色->RGB值,记住不要输出其它无关信息,现输入多个颜色:橘色,黄光色,绿色,青色,蓝光色,天蓝色,紫色,白色,粉色,粉红色,紫红色,浅绿色,棕色,橘红色,灰色,金色,浅蓝色,米色,深蓝色,咖啡色,桃红色,藏青色,紫罗兰色,深绿色,浅紫色,湖蓝色,黑色,墨绿色,银色\n",
        "max_tokens": 2000,
        "temperature": 0
    }'
