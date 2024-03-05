import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
import re

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
    

def extract_features(url):
    url=str(url)
    features = {
        "qty_dot_url": url.count('.'),
        "qty_hyphen_url": url.count('-'),
        "qty_underline_url": url.count('_'),
        "qty_slash_url": url.count('/'),
        "qty_questionmark_url": url.count('?'),
        "qty_equal_url": url.count('='),
        "qty_at_url": url.count('@'),
        "qty_and_url": url.count('&'),
        "qty_exclamation_url": url.count('!'),
        "qty_space_url": url.count(' '),
        "qty_tilde_url": url.count('~'),
        "qty_comma_url": url.count(','),
        "qty_plus_url": url.count('+'),
        "qty_asterisk_url": url.count('*'),
        "qty_hashtag_url": url.count('#'),
        "qty_dollar_url": url.count('$'),
        "qty_percent_url": url.count('%'),
        "qty_tld_url": url.count(".com") + url.count(".org") + url.count(".net"),  # Assuming common TLDs
        "length_url": len(url)
    }

    domain = re.search('//([a-zA-Z0-9.-]+)', url)
    if domain:
        domain = domain.group(1)
        domain_features = {
            "qty_dot_domain": domain.count('.'),
            "qty_hyphen_domain": domain.count('-'),
            "qty_underline_domain": domain.count('_'),
            "qty_slash_domain": domain.count('/'),
            "qty_questionmark_domain": domain.count('?'),
            "qty_equal_domain": domain.count('='),
            "qty_at_domain": domain.count('@'),
            "qty_and_domain": domain.count('&'),
            "qty_exclamation_domain": domain.count('!'),
            "qty_space_domain": domain.count(' '),
            "qty_tilde_domain": domain.count('~'),
            "qty_comma_domain": domain.count(','),
            "qty_plus_domain": domain.count('+'),
            "qty_asterisk_domain": domain.count('*'),
            "qty_hashtag_domain": domain.count('#'),
            "qty_dollar_domain": domain.count('$'),
            "qty_percent_domain": domain.count('%'),
            "qty_vowels_domain": sum(1 for char in domain if char.lower() in 'aeiou')
        }
    else:
        domain_features = {
            "qty_dot_domain": 0,
            "qty_hyphen_domain": 0,
            "qty_underline_domain": 0,
            "qty_slash_domain": 0,
            "qty_questionmark_domain": 0,
            "qty_equal_domain": 0,
            "qty_at_domain": 0,
            "qty_and_domain": 0,
            "qty_exclamation_domain": 0,
            "qty_space_domain": 0,
            "qty_tilde_domain": 0,
            "qty_comma_domain": 0,
            "qty_plus_domain": 0,
            "qty_asterisk_domain": 0,
            "qty_hashtag_domain": 0,
            "qty_dollar_domain": 0,
            "qty_percent_domain": 0,
            "qty_vowels_domain": 0
        }

    return pd.DataFrame([{**features, **domain_features}])

# # Example usage
# url = "www.facebook.com"
# data = extract_features(url)
# print(data.to_html("d.html"))

# p1=PredictPipeline()
# pvalue=p1.predict(data)
# print(pvalue)