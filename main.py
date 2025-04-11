import asyncio
import json
import csv  # Import the CSV module
from crawl4ai import (
    AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig, LLMConfig
)
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from pydantic import BaseModel

URL_TO_SCRAPE = "https://www.mumsnet.com/discover/all-in/ages-stages"

INSTRUCTION_TO_LLM = (
    "Extract all discussion posts from the page. For each post, extract the "
    "'title' and the 'content' of the initial post as separate fields."
)

class Discussion(BaseModel):
    title: str
    content: str

async def main():
    llm_config_deepseek = LLMConfig(
        provider="deepseek/deepseek-chat",
        api_token="sk-081ef2f1e604473fb3eff9eaa31dd6c9",
        temprature=0.0,
        max_tokens=800
    )

    llm_strategy = LLMExtractionStrategy(
        llm_config=llm_config_deepseek,
        schema=Discussion.model_json_schema(),
        extraction_type="schema",
        instruction=INSTRUCTION_TO_LLM,
        chunk_token_threshold=1000,
        overlap_rate=0.0,
        apply_chunking=True,
        input_format="markdown",
    )

    crawl_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS,
        process_iframes=False,
        remove_overlay_elements=True,
        exclude_external_links=True,
    )

    browser_cfg = BrowserConfig(headless=True, verbose=True)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=URL_TO_SCRAPE, config=crawl_config)

        if result.success:
            data = json.loads(result.extracted_content)
            print("Extracted items:", data)

            # Write data to CSV
            with open('extracted_data.csv', mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Title', 'Content'])  # Write the header
                for item in data:
                    if not item.get('error', True):  # Check if there's no error
                        writer.writerow([item['title'], item['content']])
            
            llm_strategy.show_usage()
        else:
            print("Error:", result.error_message)

if __name__ == "__main__":
    asyncio.run(main())