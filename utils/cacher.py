import os
import json
import uuid
from typing import Any, Dict, Union, Literal, Optional


def generate_filename(text: Optional[str] = None) -> str:
    """Creates a valid filename from a string by removing charcaters like: 
    `://`, `/`, `-`, ` ` from the given string. Or generates an unique name if `text` is `None`.
    """
    if not text:
        return str(uuid.uuid4())
    
    _new_text = text
    _replace_chars = ['://', '/', '-', ' ', ';', '\'', '"', ':']

    for char in _replace_chars:
        _new_text = _new_text.replace(char, '_')
    return _new_text


class Cacher:
    cache_dir = 'test_data/cache'
    replace_chars = ['://', '/', '-', ' ']

    def __init__(self,
        url: str,
        data_format: Literal['text', 'json', None] = 'text',
        cache_dir: Optional[str] = None
    ) -> None:
        self._url: str = url
        self._cache_file = None
        self._base_filename: str = None
        self._cache_format: str = data_format

        if cache_dir:
            self.cache_dir = cache_dir

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    @property
    def cache_file(self) -> str:
        if not self._cache_file:
            name, ext = os.path.splitext(self._url)

            if not self._cache_format:
                if ext:
                    self.cache_format = ext[1:]

            for char in self.replace_chars:
                name = name.replace(char, '_')
            
            self._cache_file = os.path.join(
                self.cache_dir, 
                name + '.' + self.cache_format
            )

            self._base_filename = name
        
        return self._cache_file
    
    @property
    def cache_format(self) -> str:
        return self._cache_format
    
    @cache_format.setter
    def cache_format(self,
        data_format: Literal['text', 'json']
    ) -> None:
        self._cache_format = data_format

    def base_filename(self) -> str:
        return self._base_filename
    
    def file_extension(self) -> str:
        return '.' + self._cache_format

    def save_cache(self,
        content: Any,
        json_config: Optional[Dict[str, Any]] = {},
    ) -> None:
        
        if not json_config:
            json_config = { 'indent': 4 }
        
        with open(self.cache_file, 'w') as cfile:
            if self.cache_format == 'json':
                json.dump(content, cfile, **json_config)
            else:
                cfile.write(content)
    
    def load_cache(self,
        json_config: Optional[Dict[str, Any]] = {}
    ) -> Union[str, Any]:
        data: Any

        with open(self.cache_file, 'r') as cfile:
            if self.cache_format == 'json':
                data = json.load(cfile, **json_config)
            else:
                data = cfile.read()
        
        return data
    
    def cache_file_exists(self) -> bool:
        return os.path.exists(self.cache_file)



if __name__ == '__main__':
    cacher = Cacher('test_cache_file2.json', data_format='json')
    cacher.save_cache({
        'file': str(__name__.__class__),
        'date': 'N/A/yuw34tg'
    })
