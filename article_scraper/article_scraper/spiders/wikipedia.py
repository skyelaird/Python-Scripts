import scrapy


class WikipediaSpider(scrapy.Spider):
    name = 'wikipedia'
    allowed_domains = ['en.wikipedia.org']
    start_urls = ['http://en.wikipedia.org/']

    def parse(self, response):
        pass
