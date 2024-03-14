import requests
from bs4 import BeautifulSoup
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Lock
import json


class IMDbCrawler:
    """
    put your own user agent in the headers
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
        # 'Accept': 'application/json',
    }
    top_250_URL = 'https://www.imdb.com/chart/top/'

    def __init__(self, crawling_threshold=1000):
        """
        Initialize the crawler

        Parameters
        ----------
        crawling_threshold: int
            The number of pages to crawl
        """
        self.crawling_threshold = crawling_threshold
        self.not_crawled = deque()
        self.crawled = set()
        self.added_ids = set()
        self.add_list_lock = Lock()
        self.add_queue_lock = Lock()

    def get_id_from_URL(self, URL):
        """
        Get the id from the URL of the site. The id is what comes exactly after title.
        for example the id for the movie https://www.imdb.com/title/tt0111161/?ref_=chttp_t_1 is tt0111161.

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        str
            The id of the site
        """
        parts = URL.split('/')
        index = parts.index('title')
        movie_id = parts[index + 1]
        movie_id = movie_id.split('?')[0]
        return movie_id

    def write_to_file_as_json(self):
        """
        Save the crawled files into json
        """
        # TODO
        pass

    def read_from_file_as_json(self):
        """
        Read the crawled files from json
        """
        # TODO
        with open('IMDB_crawled.json', 'r') as f:
            self.crawled = None

        with open('IMDB_not_crawled.json', 'w') as f:
            self.not_crawled = None

        self.added_ids = None

    def crawl(self, URL):
        """
        Make a get request to the URL and return the response

        Parameters
        ----------
        URL: str
            The URL of the site
        Returns
        ----------
        requests.models.Response
            The response of the get request
        """
        try:
            response = requests.get(URL, headers=self.headers)
            if response.status_code == 200:
                return response
            else:
                print(f'Failed to crawl {URL}.\nStatus code: {response.status_code}')
        except Exception as e:
            print(f'Failed to crawl {URL}. Exception: {str(e)}')
        return None

    def extract_top_250(self):
        """
        Extract the top 250 movies from the top 250 page and use them as seed for the crawler to start crawling.
        """
        # TODO update self.not_crawled and self.added_ids

    def get_imdb_instance(self):
        return {
            'id': None,  # str
            'title': None,  # str
            'first_page_summary': None,  # str
            'release_year': None,  # str
            'mpaa': None,  # str
            'budget': None,  # str
            'gross_worldwide': None,  # str
            'rating': None,  # str
            'directors': None,  # List[str]
            'writers': None,  # List[str]
            'stars': None,  # List[str]
            'related_links': None,  # List[str]
            'genres': None,  # List[str]
            'languages': None,  # List[str]
            'countries_of_origin': None,  # List[str]
            'summaries': None,  # List[str]
            'synopsis': None,  # List[str]
            'reviews': None,  # List[List[str]]
        }

    def start_crawling(self):
        """
        Start crawling the movies until the crawling threshold is reached.
        TODO: 
            replace WHILE_LOOP_CONSTRAINTS with the proper constraints for the while loop.
            replace NEW_URL with the new URL to crawl.
            replace THERE_IS_NOTHING_TO_CRAWL with the condition to check if there is nothing to crawl.
            delete help variables.

        ThreadPoolExecutor is used to make the crawler faster by using multiple threads to crawl the pages.
        You are free to use it or not. If used, not to forget safe access to the shared resources.
        """

        # help variables
        WHILE_LOOP_CONSTRAINTS = None
        NEW_URL = None
        THERE_IS_NOTHING_TO_CRAWL = None

        self.extract_top_250()
        futures = []
        crawled_counter = 0

        with ThreadPoolExecutor(max_workers=20) as executor:
            while WHILE_LOOP_CONSTRAINTS:
                URL = NEW_URL
                futures.append(executor.submit(self.crawl_page_info, URL))
                if THERE_IS_NOTHING_TO_CRAWL:
                    wait(futures)
                    futures = []

    def crawl_page_info(self, URL):
        """
        Main Logic of the crawler. It crawls the page and extracts the information of the movie.
        Use related links of a movie to crawl more movies.
        
        Parameters
        ----------
        URL: str
            The URL of the site
        """
        print("new iteration")
        # TODO
        pass

    def extract_movie_info(self, res, movie, URL):
        """
        Extract the information of the movie from the response and save it in the movie instance.

        Parameters
        ----------
        res: requests.models.Response
            The response of the get request
        movie: dict
            The instance of the movie
        URL: str
            The URL of the site
        """
        try:
            soup = BeautifulSoup(res.text, 'html.parser')
            movie['title'] = self.get_title(soup)
            movie['first_page_summary'] = self.get_first_page_summary(soup)
            movie['release_year'] = self.get_release_year(soup)
            movie['mpaa'] = self.get_mpaa(soup)
            movie['budget'] = self.get_budget(soup)
            movie['gross_worldwide'] = self.get_gross_worldwide(soup)
            movie['rating'] = self.get_rating(soup)
            movie['directors'] = self.get_director(soup)
            movie['writers'] = self.get_writers(soup)
            movie['stars'] = self.get_stars(soup)
            movie['related_links'] = self.get_related_links(soup)
            movie['genres'] = self.get_genres(soup)
            movie['languages'] = self.get_languages(soup)
            movie['countries_of_origin'] = self.get_countries_of_origin(soup)
            movie['summaries'] = self.get_summary(soup)
            movie['synopsis'] = self.get_synopsis(soup)
            movie['reviews'] = self.get_reviews_with_scores(soup)

        except Exception as e:
            print(f"Failed to extract information from {URL}. Exception: {str(e)}")

    def get_summary_link(url):
        """
        Get the link to the summary page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/plotsummary is the summary page

        Parameters
        ----------
        url: str
            The URL of the site
        Returns
        ----------
        str
            The URL of the summary page
        """
        try:
            movie_id = IMDbCrawler.get_id_from_URL(url)
            plot_summary_url = f"https://www.imdb.com/title/{movie_id}/plotsummary"
            return plot_summary_url
        except:
            print("failed to get summary link")
            return None

    def get_review_link(url):
        """
        Get the link to the review page of the movie
        Example:
        https://www.imdb.com/title/tt0111161/ is the page
        https://www.imdb.com/title/tt0111161/reviews is the review page
        """
        try:
            movie_id = IMDbCrawler.get_id_from_URL(url)
            reviews_url = f"https://www.imdb.com/title/{movie_id}/reviews"
            return reviews_url
        except:
            print("failed to get review link")
            return None

    def get_title(soup):
        """
        Get the title of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The title of the movie

        """
        try:
            json_ld_script = soup.find('script', type='application/ld+json').string
            json_data = json.loads(json_ld_script)
            title = json_data.get('name', None)
            return title
        except:
            print("failed to get title")
            return None

    def get_first_page_summary(soup):
        """
        Get the first page summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The first page summary of the movie
        """
        try:
            json_ld_script = soup.find('script', type='application/ld+json').string
            json_data = json.loads(json_ld_script)
            description = json_data.get('description', None)
            return description
        except:
            print("failed to get first page summary")
            return None

    def get_director(soup):
        """
        Get the directors of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The directors of the movie
        """
        try:
            json_ld_script = soup.find('script', type='application/ld+json').string
            json_data = json.loads(json_ld_script)
            directors = json_data.get('director', None)
            return directors
        except:
            print("failed to get director")
            return None

    def get_stars(soup):
        """
        Get the stars of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The stars of the movie
        """
        try:
            json_ld_script = soup.find('script', type='application/ld+json').string
            json_data = json.loads(json_ld_script)
            actors = json_data.get('actor', None)
            return actors
        except:
            print("failed to get stars")
            return None

    def get_writers(soup):
        """
        Get the writers of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The writers of the movie
        """
        try:
            json_ld_script = soup.find('script', type='application/ld+json').string
            json_data = json.loads(json_ld_script)
            writers = json_data.get('creator', None)
            return writers
        except:
            print("failed to get writers")
            return None

    def get_related_links(soup):
        """
        Get the related links of the movie from the More like this section of the page from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The related links of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get related links")

    def get_summary(soup):
        """
        Get the summary of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The summary of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get summary")

    def get_synopsis(soup):
        """
        Get the synopsis of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The synopsis of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get synopsis")

    def get_reviews_with_scores(soup):
        """
        Get the reviews of the movie from the soup
        reviews structure: [[review,score]]

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[List[str]]
            The reviews of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get reviews")

    def get_genres(soup):
        """
        Get the genres of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The genres of the movie
        """
        try:
            # TODO
            pass
        except:
            print("Failed to get generes")

    def get_rating(soup):
        """
        Get the rating of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The rating of the movie
        """
        try:
            # TODO
            pass
        except:
            print("failed to get rating")

    def get_mpaa(soup):
        """
        Get the MPAA of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The MPAA of the movie
        """
        try:
            script_element = soup.find('script', type='application/json', text=lambda text: 'certificate' in text)
            script_content = script_element.text if script_element else None
            mpaa = None
            if script_content:
                json_data = json.loads(script_content)
                mpaa = json_data.get('props', {}).get('pageProps', {}).get('aboveTheFoldData', {}).get('certificate',
                                                                                                       {}).get('rating',
                                                                                                               None)
            return mpaa
        except:
            print("failed to get mpaa")
            return None

    def get_release_year(soup):
        """
        Get the release year of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The release year of the movie
        """
        try:
            script_element = soup.find('script', type='application/json', text=lambda text: 'releaseDate' in text)
            script_content = script_element.text if script_element else None
            year = None
            if script_content:
                json_data = json.loads(script_content)
                year = json_data.get('props', {}).get('pageProps', {}).get('aboveTheFoldData', {}).get('releaseDate', {}).get('year', None)
            return year
        except:
            print("failed to get release year")
            return None

    def get_languages(soup):
        """
        Get the languages of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The languages of the movie
        """
        try:
            script_element = soup.find('script', type='application/json', text=lambda text: 'spokenLanguages' in text)
            script_content = script_element.text if script_element else None
            languages = None
            if script_content:
                json_data = json.loads(script_content)
                languages = json_data.get('props', {}).get('pageProps', {}).get('mainColumnData', {}).get(
                    'spokenLanguages', {}).get('spokenLanguages', None)
            return languages
        except:
            print("failed to get languages")
            return None

    def get_countries_of_origin(soup):
        """
        Get the countries of origin of the movie from the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        List[str]
            The countries of origin of the movie
        """
        try:
            script_element = soup.find('script', type='application/json', text=lambda text: 'countriesOfOrigin' in text)
            script_content = script_element.text if script_element else None
            countries = None
            if script_content:
                json_data = json.loads(script_content)
                countries = json_data.get('props', {}).get('pageProps', {}).get('aboveTheFoldData', {}).get(
                    'countriesOfOrigin', {}).get('countries', None)
            return countries
        except:
            print("failed to get countries of origin")
            return None

    def get_budget(soup):
        """
        Get the budget of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The budget of the movie
        """
        try:
            script_element = soup.find('script', type='application/json', text=lambda text: 'productionBudget' in text)
            script_content = script_element.text if script_element else None
            budget = None
            if script_content:
                json_data = json.loads(script_content)
                budget = json_data.get('props', {}).get('pageProps', {}).get('mainColumnData', {}).get('productionBudget', {}).get('budget', None)
            return budget
        except:
            print("failed to get budget")
            return None

    def get_gross_worldwide(soup):
        """
        Get the gross worldwide of the movie from box office section of the soup

        Parameters
        ----------
        soup: BeautifulSoup
            The soup of the page
        Returns
        ----------
        str
            The gross worldwide of the movie
        """
        try:
            script_element = soup.find('script', type='application/json', text=lambda text: 'worldwideGross' in text)
            script_content = script_element.text if script_element else None
            gross_worldwide = None
            if script_content:
                json_data = json.loads(script_content)
                gross_worldwide = json_data.get('props', {}).get('pageProps', {}).get('mainColumnData', {}).get(
                    'worldwideGross', {}).get('total', None)
            return gross_worldwide
            pass
        except:
            print("failed to get gross worldwide")
            return None


# testing soup extractions
def soup_extractions():
    url = "https://www.imdb.com/title/tt1160419/"  # dune
    # url = "https://www.imdb.com/title/tt4154796/"  # end game, multiple directors
    # url = "https://www.imdb.com/title/tt1832382/"  # a separation
    try:
        response = requests.get(url, headers=IMDbCrawler.headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            title = IMDbCrawler.get_title(soup)
            first_page_summary = IMDbCrawler.get_first_page_summary(soup)
            directors = IMDbCrawler.get_director(soup)
            stars = IMDbCrawler.get_stars(soup)
            writers = IMDbCrawler.get_writers(soup)

            mpaa = IMDbCrawler.get_mpaa(soup)
            release_year = IMDbCrawler.get_release_year(soup)
            languages = IMDbCrawler.get_languages(soup)

            countries_of_origin = IMDbCrawler.get_countries_of_origin(soup)
            budget = IMDbCrawler.get_budget(soup)
            gross_worldwide = IMDbCrawler.get_gross_worldwide(soup)
            print("Title:", title)
            print("First page summary : ", first_page_summary)
            print("Directors: ", directors)
            print("Stars: ", stars)
            print("Writers: ", writers)

            print("MPAA: ", mpaa)
            print("Release year: ", release_year)
            print("Languages: ", languages)
            print("Countries of origin: ", countries_of_origin)
            print("Budget: ", budget)
            print("Gross worldwide: ", gross_worldwide)
        else:
            print(f"Failed to fetch data from {url}. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def main():
    soup_extractions()
    imdb_crawler = IMDbCrawler(crawling_threshold=600)
    # imdb_crawler.read_from_file_as_json()
    imdb_crawler.start_crawling()
    imdb_crawler.write_to_file_as_json()


if __name__ == '__main__':
    main()
