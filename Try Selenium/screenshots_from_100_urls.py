from selenium import webdriver
import time
from selenium.webdriver.common.by import By
import base64
from tqdm import tqdm


all_children = []
urls = []
cookie_keys = []


def rekursive_screenshots(url="", key1='div > div >div > div > div > div> div > div', key_to_accept_cookies_request=''):
    num = 1
    # driver = webdriver.Chrome()
    driver.get(url)
    # cookies = driver.get_cookies()
    # driver.get(url)
    # for cookie in cookies:
    #     driver.add_cookie(cookie)

    # if url == "https://www.google.com":
    #     driver.find_element_by_xpath('//*[@id="L2AGLb"]/div').click()
    # if url == 'https://www.amazon.de':
    #     driver.find_element_by_xpath('/html/body/div[1]/span/form/div[2]/span[1]/span/input').click()

    time.sleep(5)
    try:
        driver.find_element_by_xpath(key_to_accept_cookies_request).click()
    except Exception :
        print('no cookies Request found')

    all_children = driver.find_elements(By.TAG_NAME, key1)

    num_children = len(all_children)
    for i in tqdm(range(num_children)):
        child = all_children[i]
        try:
            if child.size['height'] != 0 and child.size['width'] != 0:
                im_child = child.screenshot_as_base64
                imgdata = base64.b64decode(im_child)
                end_index = url.find('.de')
                file = open('web_recursive_screenshots/' + url[12:end_index] + str(i) + '.jpg', 'wb')
                file.write(imgdata)
        except Exception :
            print('there were ' + str(num) + ' screenshots failed')
            num += 1
            continue


def get_urls(url):

    driver.get(url)
    all_urls = driver.find_elements_by_tag_name('td > a')
    for i in range(50):
        _text = all_urls[i].get_attribute('text')
        text = 'https://'+_text
        urls.append(text)
    return urls


if __name__ == "__main__":
    driver = webdriver.Chrome()

    with open("cockies_xpath.txt", "r") as grilled_cheese:
        lines = grilled_cheese.readlines()
        for i in range(int(len(lines)/2)):
            cookie_keys.append(lines[2*i])

    get_urls("https://www.ehi.org/de/top-100-umsatzstaerkste-onlineshops-in-deutschland/")

    for index, url in enumerate(urls):
        print('taking screenshots from ' + str(index+1) + '. web')
        rekursive_screenshots(url=url, key_to_accept_cookies_request=cookie_keys[index])



