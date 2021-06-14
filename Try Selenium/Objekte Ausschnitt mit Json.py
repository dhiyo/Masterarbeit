from selenium import webdriver
from PIL import Image



driver = webdriver.Chrome()
driver.get('http://www.google.com')
element = driver.find_element_by_id('3406f3fc-cb56-4923-a8a1-1b13b1c18349')
location = element.location
size = element.size

driver.save_screenshot("shot.png")
x = location['x']
y = location['y']
w = size['width']
h = size['height']
width = x + w
height = y + h

im = Image.open('shot.png')
im = im.crop((int(x), int(y), int(width), int(height)))
im.save('image.png')
