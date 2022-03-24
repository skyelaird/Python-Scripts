import scrapy
import w3lib.html


class IetfSpider(scrapy.Spider):
    name = 'ietf'
    allowed_domains = ['pythonscraping.com']
    start_urls = ["http://pythonscraping.com/linkedin//ietf.html"]

    def parse(self, response):

        return{
            'number': response.xpath('//span[@class="rfc-no"]/text()').get(),
            'title' : response.xpath('//meta[@name="DC.Title"]/@content').get(),
            'date' : response.xpath('//span[@class="date"]/text').get(),
            'description' : response.xpath('//meta[@name="DC.Description.Abstract"]/@content').get(),            
            
        }

