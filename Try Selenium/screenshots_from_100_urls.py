from selenium import webdriver
from selenium.webdriver.common.by import By
import base64
import time
from tqdm import tqdm
from selenium.webdriver.chrome.options import Options


urls = []
web_list = []


def recursive_screenshots(url='', key='div'):

    driver.get(url)
    time.sleep(10)

    # all_children_from_url = list(set(driver.find_elements(By.TAG_NAME, key))
    #                              - set(driver.find_elements(By.TAG_NAME, key + '>div')))
    all_children_from_url = driver.find_elements(By.TAG_NAME, key)
    num_children = len(all_children_from_url)

    all_children = []

    print('loading all elements on ' + url)

    for i in tqdm(range(num_children)):
        child = []
        try:
            if all_children_from_url[i].size['width'] >= 5 and all_children_from_url[i].size['height'] >= 5:
                child.append(all_children_from_url[i].location)
                child.append(all_children_from_url[i].size)
                child.append(all_children_from_url[i].screenshot_as_base64)
                all_children.append(child)
            else:
                continue
        except Exception: pass

    invalid_children_indices = []

    print('Filtering duplicate images')
    for i in tqdm(range(len(all_children))):
        x1 = all_children[i][0]['x']
        y1 = all_children[i][0]['y']
        w1 = all_children[i][1]['width']
        h1 = all_children[i][1]['height']
        for j in range(len(all_children)):
            if j != i:
                x2 = all_children[j][0]['x']
                y2 = all_children[j][0]['y']
                w2 = all_children[j][1]['width']
                h2 = all_children[j][1]['height']
                if x1 > (x2 + w2):
                    continue
                if y1 > (y2 + h2):
                    continue
                if (x1 + w1) < x2:
                    continue
                if (y1 + h1) < y2:
                    continue
                col_int = abs(min(x1 + w1, x2 + w2) - max(x1, x2))
                row_int = abs(min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = col_int * row_int
                area1 = w1 * h1
                area2 = w2 * h2
                iou = overlap_area / (area1 + area2 - overlap_area)
                if iou > 0.4:
                    invalid_children_indices.append(j)

    for i in range(len(all_children)):
        if i not in invalid_children_indices:
            im_child = all_children[i][2]
            img_data = base64.b64decode(im_child)
            point_indices = url.split('.')
            file = open('D:/dataset/SSDe100/' + point_indices[1] + str(i) + '.jpg', 'wb')
            file.write(img_data)


def get_urls(parent_url):

    driver.get(parent_url)
    all_urls = driver.find_elements_by_tag_name('td > a')
    for i in range(len(all_urls)):
        _text = all_urls[i].get_attribute('text')
        # if _text.find('www.') == -1:
        #     _text = 'www.'+_text
        text = 'https://'+_text
        urls.append(text)
    return urls


def get_driver():

    chrome_options = Options()
    chrome_options.add_extension(r"C:\Users\shang\Downloads\i_don't_care_about_cookies.crx")
    my_driver = webdriver.Chrome(options=chrome_options)
    my_driver.maximize_window()
    return my_driver


if __name__ == "__main__":

    driver = get_driver()

    web_list = get_urls("https://www.ehi.org/de/top-100-umsatzstaerkste-onlineshops-in-deutschland/")
    invalid_list = [46,26,59]

    for index, url in enumerate(web_list):
        if index == 61:
            if index not in invalid_list:
                print('\nworking on ' + str(index+1) + '. web')
                recursive_screenshots(url=url)


