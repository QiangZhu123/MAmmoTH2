import requests
from bs4 import BeautifulSoup
def extract_links(html):
    """
    collect the links in the html 
    
    Args:
        html(str): the input website
    Return:
        links([str]): all the links in the webite
    """
    soup = BeautifulSoup(html, 'html.parser')
 
    links = [a['href'] for a in soup.find_all('a', href=True)]
    return links

def scrape_links(base_url, num_links):
    
    """
    According to the web construction,
    
    Args:
       base_url(str):the site
       num_links(int): how many links to save
    Return:
       unique_links(str): link strings used for subsequent code 
    """
    links = []
    page_num = 1
    
    while len(links) < num_links:
        url = f"{base_url}/questions?page={page_num}&sort=newest"
        html = fetch_website_content(url)
        if html:
            page_links = extract_links(html)
            content_links = [link for link in page_links if '/questions/' in link]
            links.extend(content_links)
            if len(content_links) == 0:
                break  
        page_num += 1
        
    # 去重和限制链接数量
    unique_links = list(set(links))[:num_links]
    return unique_links
def extract_text_from_url(url):
    """
    
    
    """
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        text = soup.get_text(separator='\n')
        
        return text
    else:
        return f"Failed to retrieve content. Status code: {response.status_code}"
def fetch_website_content(url):
    
    """
    Get the website data
    Args:
        url (str):the websites
    Return:
        response.text(text):text with lots of noise 
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  
        return response.text
    except requests.RequestException as e:
        print(f"请求出错: {e}")
        return None
def get_root_url(url):
    """
    Get the web root URL.
    
   Args:
       url(str): websites
    Return:
        root_url(str ): root URL
    """
    parsed_url = urlparse(url)
    root_url = urlunparse((
        parsed_url.scheme,  
        parsed_url.netloc, 
        '',                
        '',              
        '',             
        ''               
    ))
    return root_url

def merge_data(data, merged_data):
    """
    Add the merged_data into the data. Not be used 
    
    Args:
        data (datasets.dataset.Dataset):HuggingFace dataset object
        merged_data ([dict]):add everyone in dataset
    Returns:
        data(datasets.dataset.Dataset):HuggingFace dataset object
    
    """
    
    for item_data in merged_data:
        data = data.add_item(item_data)
    return data

def recall_tokens(base_data,model,size):
    """
    Using the model to classify the base_data's item , if the item is relevant document,save it's root URL, and accumulate tokens,until the tokens size meets the requirements.
    Finally,remove items that do not meet the required nums.
    
    Args:
        base_data (datasets.dataset.Dataset):HuggingFace dataset object
        model ( ):fastText
        size ( int ): depend on the stage,100B or 40B.
    Return:
        urls_count (dict):all the root URL counts over 1000 
    """
    tokens_counts = 0
    urls_count={}#count the documents

    for i in range(len(base_data)):
        root_url = get_root_url(base_data[i]['url'] )
        if root_url not in urls_count:
            urls_count[root_url]=0
        if model.predict(base_data[i]['text'])==1:
            urls_count[root_url]+=1
            tokens_counts+=len(base_data[i]['text'])
        if tokens_counts>= size:
            break
    for url_root in urls_count:
        if urls_count[url_root]<1000:
            urls_count.pop(url_root)
    return urls_count

def recall_domains(llm,urls_count):
    """
    Using the LLM to recall the domains,find the domains that have more knowledge in math, science, engineering,etc.
    Need to try suit prompt and good example to help model return answer.

    Args:
        llm(Module): the LLM model,need API key
        urls_count(dict): the root URL that may have more useful knowledge.
    Return:
        used_domains(set): if a root URL may contain instruction data,keep it in set.

    """
    
    prompt = PromptTemplate.from_template("""
        Given the root URL is {url},please tell me if it may contain instruction data,
        if it don't have math, science, engineering ,return  "there are no question-answer pairs in the url".
    """)
    used_domains=set()
    for url in urls_count:
        if not "there are no relative information in the url"  in llm.invoke(prompt.format(url=url)):
            used_domains.add(url)
    return used_domains

def clean_and_filter(url_list):
    """
    The parsed documents have many spaces and \n in them,need to delete to short the length
    
    Args:
       url_list([str]): all the url path.
    Return:
       results([list]): short and clean for the inference.
    """
    results=[]
    for url in url_list:
        text = extract_text_from_url(url)
        cleaned_text = re.sub('\n{2,}','',text)
        cleaned_text = re.sub('\s{2,}','',cleaned_text)  
        results.append(cleaned_text)

    return results