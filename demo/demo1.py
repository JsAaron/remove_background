from PIL import Image, ImageDraw
from util import getImagePath
imageURl = getImagePath("1.jpg")

def checkForMatch(color1, color2):
    for a in range(0, 6):
        for b in range(0, 6):
            for c in range(0, 6):
                color2 = color2[0] + a, (color2[1]- 3) + b, color2[2] + c
                if color1 == color2:
                    return "1"
    

def ChromaKey(image):
    backImage = Image.open('back.jpg')
    imageCopy = image.copy()
    width = imageCopy.size[0]
    height = imageCopy.size[1]
    color = (0, 255, 0) #green
    color2 = (0, 255, 1)
    color3 = (0, 255, 2)
    
    newImage = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(newImage)
    for row in range(0, height):
        for col in range(0, width):
            pix = imageCopy.getpixel((col,row))
            putColor = backImage.getpixel((col,row))
            xy = col, row
            #checky = checkForMatch(pix, color)
            #if check == 1:
            if pix == color or pix == color2 or pix == color3:
                newImage.putpixel((xy), putColor)
            #if check != 1:
            if pix != color:
                newImage.putpixel((xy), pix)
            
                
    newImage.show()
    
original = Image.open(imageURl)
modified = ChromaKey(original)