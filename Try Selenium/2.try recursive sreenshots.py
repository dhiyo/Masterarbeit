from selenium import webdriver
from PIL import Image
from selenium.webdriver.common.by import By
import base64

all_children = []


def rekursive_screenshot_by_selector(url="https://www.google.com", selector='div'):  #去掉xpath项 加上递归模式参数

    driver = webdriver.Chrome()
    driver.get(url)
    if url == "https://www.google.com":
        driver.find_element_by_xpath('//*[@id="L2AGLb"]/div').click()

    all_children = driver.find_elements(By.CSS_SELECTOR, selector)

    num_children = len(all_children)
    for i in range(num_children):
        child = all_children[i]
        if child.rect['height'] != 0 and child.rect['width']:
            im_child = child.screenshot_as_base64
            imgdata = base64.b64decode(im_child)
            file = open(str(i)+'.jpg', 'wb')
            file.write(imgdata)
            # TODO：
            # new_selector = child.tag_name
            # return rekursive_screenshot_by_selector(url, new_selector)


# def screenshot_for_element(url, id):
#     driver = webdriver.Chrome()
#     driver.get(url)
#     # find Element under the element
#     element = driver.find_element_by_id(id)
#     location = element.location
#     size = element.size
#
#     driver.save_screenshot(id)
#
#     x = int(location['x'])
#     y = int(location['y'])
#     width = int(size['width'])
#     height = int(size['height'])
#
#     im_crop(id, x, y, width, height)
#
#
# def im_crop(filename, x, y, width, height):
#     im = Image.open(filename)  # uses PIL library to open image in memory
#     im = im.crop((x, y, x + width, y + height))  # defines crop points
#     im.save(filename)  # saves new cropped image
if __name__ == "__main__":
    rekursive_screenshot_by_selector(url='https://campus.studium.kit.edu/events/timetable.php')


