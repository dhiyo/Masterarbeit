from selenium import webdriver
import time
from PIL import Image


DRIVER_PATH = 'C:/Users/shang/Documents/yueqi_ws/Masterarbeit/Try Selenium/chromedriver'
driver = webdriver.Chrome(executable_path=DRIVER_PATH)
driver.get('https://stackoverflow.com/questions/13832322/how-to-capture-the-screenshot-of-a-specific-element-rather-than-entire-page-usin/51517606#51517606')
driver.find_elements_by_xpath('/html')[0]

driver.save_screenshot('screenshot.png')
Img = Image.open('screenshot.png')
Img.show()