from selenium import webdriver
import time
from selenium.webdriver.common.by import By
import base64
from tqdm import tqdm
from selenium.webdriver.chrome.options import Options



def get_driver():

    chrome_options = Options()
    chrome_options.add_extension(r"C:\Users\shang\Downloads\i_don't_care_about_cookies.crx")
    my_driver = webdriver.Chrome(options=chrome_options)
    my_driver.maximize_window()
    return my_driver



def find_element(url='', depth=1, key='div'):

    driver.get(url)
    all_children_from_url = list(set(driver.find_elements(By.TAG_NAME, key))
                                 - set(driver.find_elements(By.TAG_NAME, key + '>div')))
    print(len(all_children_from_url))
    for i in range(len(all_children_from_url)):
        try:
            im_child = all_children_from_url[i].screenshot_as_base64
            img_data = base64.b64decode(im_child)
            end_index = url.find('.de')
            file = open('otto/' + str(depth) + url[12:end_index] + str(i) + '.jpg', 'wb')
            file.write(img_data)
        except Exception: continue

    if len(all_children_from_url) != 0:
        return find_element(url, depth+1, key+'> div')


if __name__ == "__main__":

    driver = get_driver()

    find_element('https://www.otto.de/')

