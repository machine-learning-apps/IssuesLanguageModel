from bs4 import BeautifulSoup as bs
import bs4
import mistune
from timeout_decorator import TimeoutError, timeout
from IPython.display import display, Markdown
from fastai.core import compose, listify, partial
from fastai.text.transform import fix_html, SpacyTokenizer, defaults, partition_by_cores
from typing import List, Union, Callable
from urllib3.util import parse_url 
import re
import regex
from textacy.preprocess import preprocess_text, normalize_whitespace
from textacy.text_utils import detect_language

# initialize markdown parser
markdown = mistune.Markdown()

class md:
    "class that organizes functions that can cleanup a namespace"
    @staticmethod
    def parse(x:str) -> bs4.BeautifulSoup:
        
        # find & replace html, which can break things (non-greedy)
        x = re.sub(r'<.+?>.+?</.+?>|<[a-zA-Z]{1,}.*?>', 'xxxhtml', x, re.DOTALL)
        
        #because former html replacement was non-greedy dedupe html marker
        x = re.sub('(xxxhtml(xxxlnbrk)?(\s)?)+', ' xxxhtml ', x)
        
        # fix the linebreak issue from BigQuery
        x = re.sub(r'xxxlnbrk( +)?', '\n', x)
       
        @timeout(1)
        def timed_parse(x):
            try:
                return bs(markdown(x), features="html5lib")
            
            except TimeoutError:
                return bs(markdown('xxxunabletoparse'), features="html5lib")
            
        return timed_parse(x)
    
    @staticmethod
    def prepend(fldname:str, tag:Union[List[str], str], soup:bs4.BeautifulSoup) -> bs4.BeautifulSoup:
        for tag in soup.find_all(listify(tag)):
            if tag.text.strip() or tag.name == 'hr':
                tag.insert(0, fldname+' ')
        return soup
    
    @staticmethod
    def enclose(bfldname:str, efldname:str, tag:Union[List[str], str], nlines:int, soup:bs4.BeautifulSoup) -> bs4.BeautifulSoup:
        """Helper function for when you want to add a beginning and ending marker to text."""
        for tag in soup.find_all(listify(tag)):
            
            # preview the text inside an enclosure show nlines of beginning and nlines of the end.
            text_lines = tag.text.split('\n')
            if len(text_lines) <= nlines * 2:
                newstr = tag.text
            else:
                newstr = '\n'.join(text_lines[:nlines] + text_lines[-nlines:])
                
            tag.string = newstr
            
            # add the values of the class attributes, if exist
            tag.insert(0, bfldname + ' ' + (' '.join(tag['class']) if 'class' in tag.attrs else '') + ' ')
            
            # insert ending tag with/without space depending if last char is \n
            if tag.text[-1] == '\n':
                tag.append(efldname)
            else:
                tag.append(' ' + efldname)
        return soup
    
    @staticmethod
    def lst(soup:bs4.BeautifulSoup) -> bs4.BeautifulSoup:
        "annotate list elements <ul> and <ol>"
        for tag in soup.find_all(['ul', 'ol']):
            # clear all the artifacts that are in lists and replace with text.
            text = 'xxxlistB ' + tag.getText() + 'xxxlistE'
            tag.string = text.strip()
        return soup
    
    @staticmethod
    def tbl(soup:bs4.BeautifulSoup) -> bs4.BeautifulSoup:
        "annotate table elements <table> only keeping information from header rows"
        for tag in soup.find_all('table'):
            # empty string if there are no table headers.
            text = ''
            if tag.thead:
                text = 'xxtbl ' + '|'.join([x.getText() for x in tag.thead.find_all('th')])
            tag.string = text
        return soup
    
    @staticmethod
    def img(soup:bs4.BeautifulSoup) -> bs4.BeautifulSoup:
        for tag in soup.find_all('img'):
            tag.insert(0, 'xxximg ')
            if 'alt' in tag.attrs:
                tag.insert(1, tag['alt'])
            if 'src' in tag.attrs:
                tag.append(' xxximgf ' + tag['src'].split('.')[-1])
        return soup
    
    @staticmethod
    def lnk(soup:bs4.BeautifulSoup) -> bs4.BeautifulSoup:
        for tag in soup.find_all('a'):
            if 'href' in tag.attrs:
                try:
                    tag.append(' xxxlnkhb ' + parse_url(tag['href']).host + ' xxxlnkhe')
                except:
                    pass
            if 'title' in tag.attrs:
                tag.append(' xxxlnktb ' + tag['title'] + 'xxxlnkte')
        return soup
    
    @staticmethod
    def get_text(soup:bs4.BeautifulSoup) -> str:
        "get the raw text"
        text = soup.getText()
        #translate newlines back from BigQuery
        text = re.sub(r'\n\n+', '\n', text)
        #translate double quotes back from BigQuery
        text = re.sub(r'xxxdblqte', ' \" ', text)
        return normalize_whitespace(text)
    
    @staticmethod
    def sym(text:str) -> str:
        """generalize symbols such as urls, emails, phone numbers and filepaths to generic tokens."""
        text = preprocess_text(text, 
                               fix_unicode=True, 
                               no_urls=True, 
                               no_emails=True, 
                               no_phone_numbers=True,
                               no_accents=True)
        
        # generalize file paths
        file_path_regex = r'C:(\\\\\S+){2,}|(/\S+){2,}|[Cc]:\\\w+(\\[0-9a-zA-Z_\-]+)+'
        text = re.sub(file_path_regex, ' xxxfilepath ', text)
        
        # generalize @ mentions
        at_mention_regex = r'\W@\w+'
        text = re.sub(at_mention_regex, ' xxxatmention ', text)
        
        # get date/time
        text = re.sub(r'\d+[-/]\d+[-/]\d+(.{0,2})?(\d+:\d+:\d+)', ' xxxdatetm ', text)
        
        # strings that have >=4 dots w/o any whitespace in between
        text = re.sub(r'(\S+\.\S+){4,}', 'xxunk', text)
        
        # things that look like IP addresses
        text = re.sub(r'\d+\.\d+.\d+\.\d+', 'xxunk', text)
        
        # long strings or numbers
        text = re.sub(r'\S{30,}|\d{6,}', 'xxunk', text)
        
        # generalize json
        json_regex = r'\{(?:[^{}]|(?R))*\}'
        text = regex.sub(json_regex, ' xxxjson ', text)
        
        return text
            
    ### transformations that are the same from factory functions
    # large headers: h1
    hL =   partial(prepend.__func__, 'xxxhl', 'h1')
    # medium headers: h2, h3
    hM =   partial(prepend.__func__, 'xxxhm', ['h2', 'h3'])
    # small headers: h4, h5, h6
    hS =   partial(prepend.__func__, 'xxxhs', ['h4', 'h5', 'h6'])
    # code blocks
    code = partial(enclose.__func__, ' xxxcdb ', ' xxxcde ', 'code', 2)
    # paragraph blocks (plain text)
    txt =  partial(prepend.__func__, '', 'p')
    # block quotes
    bqt =  partial(enclose.__func__, 'xxxqb', 'xxxqe', 'blockquote', 3)
    # strikethrough
    st =   partial(enclose.__func__, 'xxxdelb', 'xxxdele', 'del', 1)
    # horizontal rule
    hr =   partial(prepend.__func__, 'xxxhr', 'hr')
    

transform_pre_rules = [md.parse, md.hL, md.hM, md.hS, md.lst, md.bqt, 
                       md.code, md.tbl, md.st, md.txt, md.lnk, md.img, 
                       md.hr, md.get_text, md.sym] + defaults.text_pre_rules