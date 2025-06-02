import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
try:
    import spacy
    try:
        nlp = spacy.load("uk_core_news_sm")
    except:
        print("Ukrainian model not available. Using alternative approach.")
        nlp = None
except:
    print("Spacy not available. Using alternative approach.")
    nlp = None

# Define constants
TIRE_ATTRIBUTES = {
    'зчеплення_сухий_асфальт': ['зчеплення сух', 'зчепл асфальт', 'тяга на сухому', 'сцепл асфальт', 'тягов на сух', 'зчепл на сух', 'сух асфальт', 'сухе покритт', 'зчепл сухому'],
    'зчеплення_волога_дорога': ['зчеплення волог', 'зчепл мокр', 'тяга на мокр', 'зчепл на волог', 'сцепл дощ', 'тягов на мокр', 'аквапланування', 'мокр асфальт', 'волог покритт', 'дощов погод', 'слизьк дорог'],
    'зчеплення_сніг': ['зчеплення сніг', 'зчепл сніж', 'тяга на сніг', 'сцепл зим', 'тягов взим', 'сніж дорог', 'снігов покрив', 'засніжен дорог'],
    'зчеплення_лід': ['зчеплення лід', 'зчепл лід', 'тяга на льод', 'сцепл лід', 'тягов на льод', 'ожелед', 'льодян покрив', 'заморожен дорог'],
    'гальмування': ['гальмів', 'гальм', 'гальмівн', 'гальмув', 'зупинка', 'зупин', 'гальмівний шлях', 'зупинн шлях', 'гальмівн властив', 'ефектив гальмув'],
    'зносостійкість_протектор': ['знос протектор', 'зносост протектор', 'довговічн протектор', 'міцн протектор', 'протектор стійк', 'протектор зношу', 'глибин протектор', 'стиран протектор'],
    'зносостійкість_загальна': ['знос', 'зносост', 'довговічн', 'міцн', 'зношув', 'ресурс', 'служб', 'тривал', 'термін служб', 'кілометраж', 'пробіг', 'витривал'],
    'протектор_дизайн': ['протектор дизайн', 'малюнок протектор', 'протектор малюн', 'рисунок протектор', 'протектор форм', 'візерунок', 'канавк'],
    'шум_рівень': ['шум', 'гучн', 'гуркіт', 'гуркот', 'шумн', 'децибел', 'звук', 'тих', 'безшумн'],
    'шум_характер': ['шум характер', 'шум тип', 'звук рівномірн', 'шум рівномірн', 'гудін', 'свист', 'шелест', 'гул'],
    'комфорт_їзда': ['комфорт їзд', 'м\'як їзд', 'жорстк їзд', 'плавн хід', 'комфорт під час їзд', 'комфортн їзд', 'м\'якіст ходу', 'плавніст'],
    'комфорт_вібрації': ['вібрац', 'тряск', 'трясе', 'вібрує', 'трясіння', 'коливан', 'тремтін'],
    'ціна_якість': ['ціна якість', 'співвідношення цін', 'співвіднош варт', 'варт за гроші', 'за свої гроші', 'цін/як', 'ціна/якіст'],
    'ціна': ['ціна', 'цін', 'варт', 'дорого', 'дешев', 'бюджет', 'доступн', 'коштує', 'коштував', 'дорогі', 'вартіст'],
    'волога_дорога_загальне': ['волог', 'мокр', 'дощ', 'калюж', 'після дощу', 'мокр покритт'],
    'сніг_глибокий': ['глибок сніг', 'заметіл', 'кучугур', 'снігов полон', 'сніжн замет', 'високий сніг'],
    'сніг_укатаний': ['укатан сніг', 'утрамбован сніг', 'спресован сніг', 'наїждж', 'твердий сніг'],
    'лід_ожеледь': ['лід', 'льод', 'ожелед', 'обмерзл', 'слизьк', 'льодян'],
    'паливна_ефективність': ['палив', 'економ палив', 'витрат палив', 'ефектив витрат', 'економічн', 'опір коченню'],
    'економічність': ['економ', 'заощадж', 'витрат', 'економн', 'бережлив', 'ощадлив'],
    'управління_маневреність': ['управлін', 'керован', 'маневр', 'маневрен', 'поворот', 'віраж', 'маневреніст'],
    'управління_стабільність': ['стабільн', 'стійк', 'курсов стійк', 'прямолінійн', 'стабіл рух', 'тримання курсу'],
    'управління_точність': ['точн керув', 'точн управл', 'керм чутлив', 'відгук на кермо', 'реакц на поворот', 'чітк керуван'],
    'сезонність_літо': ['літн', 'літо', 'спек', 'спекотн', 'висок температур', 'тепл', 'жарк'],
    'сезонність_зима': ['зимов', 'зим', 'зимн', 'холод', 'низьк температур', 'мороз', 'зимн умов'],
    'сезонність_всесезонність': ['всесезон', 'універсал', 'і зим і літ', 'всепогодн', 'цілорічн'],
    'міцність_боковини': ['боков', 'бічн', 'боковин', 'боковин міцн', 'бічн стінк', 'стінк', 'міцність боків'],
    'стійкість_до_пошкоджень': ['пошкодж', 'проколи', 'порізи', 'стійк до пошкодж', 'удар', 'камін', 'скло', 'міцніст'],
    'посадка_кріплення': ['посадк', 'кріпл', 'монтаж', 'балансуван', 'дисбаланс', 'встановлен'],
    'поведінка_навантаженні': ['навантаж', 'вантаж', 'багаж', 'під вагою', 'важк', 'під навантажен'],
    'якість_виробництва': ['якіст', 'виробн', 'збірк', 'виготовл', 'завод', 'якіст продукц'],
    'дизайн_зовнішній': ['дизайн', 'вигляд', 'естетик', 'естетичн', 'зовнішн', 'краса'],
    'бренд_довіра': ['бренд', 'виробник', 'фірм', 'репутац', 'марк', 'довір', 'надійн виробник'],
    'країна_виробництва': ['країн', 'виробн в', 'зроблен в', 'походжен', 'вироблен', 'країна походжен'],
    'гарантія': ['гарант', 'повернен', 'заміна', 'гарантійн', 'сервіс', 'гарантійн терм'],
    'ціна_доставки': ['доставк', 'пересил', 'транспорт', 'вартіст доставк', 'ціна доставк'],
    'шиповані': ['шип', 'шипован', 'шиповк', 'шипи', 'з шипами', 'шипи металев'],
    'бездоріжжя': ['бездоріжж', 'оффроуд', 'грунт', 'грунтівк', 'польов', 'ґрунтов', 'пісок', 'болот', 'багн', 'бездоріжн умов']
}
POSITIVE_WORDS = [
    # Базові позитивні слова
    'добре', 'добрий', 'чудовий', 'відмінний', 'хороший', 'надійний', 'якісний', 
    'відмінно', 'прекрасний', 'задоволений', 'рекомендую', 'вартий', 'люблю', 
    'варті', 'приємний', 'впевнений', 'комфортний', 'плавний', 'тихий', 'економний',
    
    # Додаткові позитивні
    'бездоганний', 'неперевершений', 'фантастичний', 'ідеальний', 'супер', 'задоволення',
    'неймовірний', 'вражаючий', 'перевершив очікування', 'перевершує', 'кращий',
    'варто', 'вигідний', 'раджу', 'рекомендував би', 'топовий', 'першокласний',
    'взяв би знову', 'купив би знову', 'зручний', 'витривалий',
    'безпечний', 'тривалий', 'ефективний', 'продуктивний', 'потужний', 'вартісний',
    'витрачені гроші', 'не шкодую', 'задоволений вибором', 'оптимальний', 'чіткий',
    'швидкий', 'легкий', 'приємно здивований', 'радий', 'неперевершено', 'досконалий',
    'збалансований', 'стабільний', 'надійно', 'безшумний', 'м\'яко', 'легко',
    'чудово тримає', 'блискуче справляється', 'витримує', 'стійкий', 'точний',
    'передбачуваний', 'впевнено', 'вражаюче', 'захоплюючий', 'захоплює', 'вражає',
    
    # Слова з контексту шин
    'тримає', 'справляється', 'працює', 'служить', 'витримав', 'дотримав', 'достойний'
]
NEGATIVE_WORDS = [
    # Базові негативні
    'погано', 'поганий', 'жахливий', 'гірший', 'розчарований', 'проблема', 'незадоволений',
    'недолік', 'недостатній', 'шумний', 'дорогий', 'неякісний', 'слабкий', 'ламається', 
    'непрактичний', 'ненадійний', 'зношується', 'зносився', 'розчарував', 'дефект', 'шкодую',
    
    # Детальніші негативні
    'огидний', 'жалкую', 'марно витрачені', 'даремно', 'провал', 'помилка', 'не рекомендую',
    'не варто', 'не радив би', 'пожалів', 'розчарування', 'неприємно здивований', 'недоліки',
    'брак', 'бракований', 'неякісно', 'зіпсований', 'не працює', 'проблемний', 'поганої якості',
    'відвертий непотріб', 'гірше нікуди', 'неприпустимий', 'неприйнятний', 'підвів', 
    'не витримує', 'не справляється', 'не відповідає', 'обман', 'розвалюється',
    'швидко зношується', 'небезпечний', 'ризикований', 'нестабільний', 'некерований',
    'заносить', 'неконтрольований', 'некомфортний', 'жорсткий', 'гучний', 'галасливий',
    'дратує', 'втомлює', 'нервує', 'незручний', 'важкий', 'грубий', 'повільний',
    'не тримає', 'ковзає', 'сковзає', 'буксує', 'не справляється', 'витратний',
    'дорогий', 'переоцінений', 'завищена ціна', 'не вартий', 'не відпрацьовує',
    
    # Контекстуальні негативні для шин
    'слизький', 'ковзкий', 'нестійкий', 'непередбачуваний', 'аварійний'
]
NEGATIONS = [
                    'не', 'ні', 'ані', 'ніколи', 'нізащо', 'нічого', 'жодного', 'жодної', 
                    'без', 'відсутній', 'немає', 'нема', 'бракує', 'ніякого', 'ніяк'
]       # List of negation words

INTENSIFIERS = [
    'дуже', 'надзвичайно', 'неймовірно', 'занадто', 'абсолютно', 'повністю',
    'цілком', 'особливо', 'надто', 'значно', 'набагато', 'сильно', 'вкрай',
    'безмежно', 'шалено', 'жахливо'
]

DIMINISHERS = [
    'трохи', 'дещо', 'злегка', 'частково', 'ледь', 'майже', 'не дуже', 'не надто',
    'доволі', 'відносно', 'порівняно', 'помірно', 'незначно', 'мінімально'
]

class EnhancedTireAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Розширений аналізатор відгуків про вантажні шини")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        self.root.state('zoomed')  # For Windows
        
        # Variables for filters and data
        self.df = None
        self.file_path = None
        self.word_freq = None
        self.filtered_df = None
        self.filter_vars = {}
        self.filter_widgets = {}
        self.sentiment_mode = tk.StringVar(value="normal")
        self.color_by_ratings = tk.BooleanVar(value=True)  # Default to coloring by sentiment
        self.color_by_sentiment = tk.BooleanVar(value=True)
        self.analyzed_df = None  # Will store sentiment analysis results
        self.sentiment_data = {}  # Will store detailed sentiment data
        
        # Initialize the variable before it's used
        self.max_words_var = tk.IntVar(value=100)
        self.colormap_var = tk.StringVar(value="viridis")
        
        # Set up styles and stopwords
        self.setup_styles()
        self.setup_stopwords()
        
        # Create UI
        self.create_ui()
        self.setup_event_handlers()

        # Try to find the data file in the current directory
        self.file_path = None
    
    # 1. UTILITY METHODS
    def setup_styles(self):
        """Set up styles for the application"""
        style = ttk.Style()
        
        # Main buttons
        style.configure('TButton', font=('Arial', 10))
        style.configure('Big.TButton', font=('Arial', 11))
        
        # Accent buttons
        style.configure('Primary.TButton', font=('Arial', 11, 'bold'))
        
        # Headers
        style.configure('TLabel', font=('Arial', 10))
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        
        # Frames
        style.configure('TLabelframe', borderwidth=2)
        style.configure('TLabelframe.Label', font=('Arial', 11, 'bold'))
    
    def setup_stopwords(self):
        """Set up stopwords for Ukrainian language"""
        # Download necessary resources for Ukrainian language
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        try:
            self.ukrainian_stopwords = set(stopwords.words('ukrainian'))
        except:
            # If Ukrainian stopwords are not available in NLTK, define our own set
            self.ukrainian_stopwords = {'і', 'в', 'на', 'з', 'та', 'до', 'що', 'не', 'є', 'як', 
                                       'це', 'за', 'від', 'по', 'у', 'я', 'ти', 'він', 'вона', 
                                       'воно', 'ми', 'ви', 'вони', 'мій', 'твій', 'свій', 'наш', 
                                       'ваш', 'їх', 'цей', 'той', 'такий', 'який', 'де', 'коли', 
                                       'чому', 'тому', 'бо', 'щоб'}

        # Add additional stopwords
        additional_stopwords = {'але', 'вже', 'для', 'при', 'після', 'які', 'навіть', 'якість', 
                               'шини', 'шин', 'дуже', 'також', 'тис', 'всі', 'най', 'нашої', 
                               'ніж', 'тих', 'тому', 'краще', 'кращою', 'дорозі', 'дороги', 
                               'км', 'год', 'роботи'}

        self.all_stopwords = self.ukrainian_stopwords.union(additional_stopwords)
    
    def normalize_columns(self):
        """Normalize column names in the file"""
        # Check for BOM in first column name and remove it
        if self.df.columns[0].startswith('\ufeff'):
            self.df.columns = [self.df.columns[0].replace('\ufeff', '')] + list(self.df.columns[1:])
        
        # Mapping for possible column names
        rating_columns = ['Рейтинг', 'Оцінка', 'Rating', 'Rate', 'Балл', 'Оценка', 'Бал']
        comment_columns = ['Коментар', 'Відгук', 'Комментарий', 'Отзыв', 'Comment', 'Review', 'Text']
        brand_columns = ['Марка_шини', 'Марка', 'Бренд', 'Brand', 'Make', 'Tire_Brand', 'Виробник']
        date_columns = ['Дата', 'Дата_відгуку', 'Date', 'Review_Date']
        
        # Check and rename columns - explicitly handling the date column
        columns_map = {}
        date_column = None
        comment_column = None
        
        # First, identify key columns (Ratings and Brand)
        for col in self.df.columns:
            col_lower = col.lower()
            
            # Skip already correctly named columns
            if col in ['Рейтинг', 'Марка_шини']:
                continue
                
            # Check if column looks like a rating
            if any(rating_name.lower() in col_lower for rating_name in rating_columns):
                columns_map[col] = 'Рейтинг'
            # Check if column looks like a brand
            elif any(brand_name.lower() in col_lower for brand_name in brand_columns):
                columns_map[col] = 'Марка_шини'
        
        # Now handle date and comment columns specifically to avoid conflicts
        for col in self.df.columns:
            col_lower = col.lower()
            
            # Skip already mapped columns
            if col in columns_map.keys() or col in ['Рейтинг', 'Марка_шини']:
                continue
                
            # Check if column looks like a date
            if any(date_name.lower() in col_lower for date_name in date_columns):
                date_column = col
            # Check if column looks like a comment
            elif any(comment_name.lower() in col_lower for comment_name in comment_columns):
                comment_column = col
        
        # Make sure we don't rename date column to 'Коментар'
        if date_column:
            columns_map[date_column] = 'Дата_відгуку'
        
        if comment_column:
            columns_map[comment_column] = 'Коментар'
        
        # Rename found columns
        if columns_map:
            self.df.rename(columns=columns_map, inplace=True)
                
        # Check if there's a rating column, and if not - create random ones
        if 'Рейтинг' not in self.df.columns:
            messagebox.showinfo("Інформація", "У файлі не знайдено колонку з оцінками. Створюємо випадкові оцінки для демонстрації.")
            self.df['Рейтинг'] = np.random.uniform(1, 5, size=len(self.df))
            self.df['Рейтинг'] = self.df['Рейтинг'].round(1)
            
        # If there's no brand column, create a dummy one
        if 'Марка_шини' not in self.df.columns:
            brands = ['Bridgestone', 'Continental', 'Michelin', 'Goodyear', 'Dunlop', 
                    'Nokian', 'Pirelli', 'Toyo', 'Yokohama', 'Hankook']
            self.df['Марка_шини'] = np.random.choice(brands, size=len(self.df))
            
        # If there's no comment column, we need actual comments
        if 'Коментар' not in self.df.columns:
            messagebox.showinfo("Інформація", "У файлі не знайдено колонку з коментарями. Створюємо тестові коментарі.")
            sample_comments = [
                "Дуже надійні шини з чудовою тривалістю служби.",
                "Хороший баланс між керованістю та зносостійкістю.",
                "Неймовірна зносостійкість! Працюю на далеких маршрутах і вони не підводять.",
                "Відмінно тримає дорогу в дощову погоду.",
                "Чудова шина для важких вантажівок, добре тримає на будь-якій поверхні.",
                "Хороше зчеплення на мокрому асфальті.",
                "Трохи шумні на високих швидкостях, але відмінно тримають дорогу.",
                "Ідеальне співвідношення ціни та якості для довгих перевезень.",
                "Довго служать, але на слизькій дорозі можуть бути кращі варіанти.",
                "Забезпечують плавний хід і зменшують витрату палива."
            ]
            self.df['Коментар'] = np.random.choice(sample_comments, size=len(self.df))
    
    # 2. TEXT ANALYSIS METHODS
    def custom_word_tokenize(self, text):
        """Tokenize text with Ukrainian language support"""
        if not isinstance(text, str):
            return []
        
        # Convert to lowercase first
        text = text.lower()
        
        # Ukrainian-specific pattern: includes apostrophe for words like "не'мати"
        # and hyphen for compound words like "інтернет-магазин"
        ukrainian_pattern = r"\b[а-яії'`ʼ-]+\b|\b[a-z'`ʼ-]+\b|\b\d+\b"
        tokens = re.findall(ukrainian_pattern, text)
        
        # Filter out single characters and clean tokens
        cleaned_tokens = []
        for token in tokens:
            # Remove leading/trailing punctuation
            cleaned_token = re.sub(r"^['-]+|['-]+$", "", token)
            if len(cleaned_token) > 1:  # Keep only words longer than 1 character
                cleaned_tokens.append(cleaned_token)
        
        return cleaned_tokens
    
    def extract_context(self, text, keywords, window_size=5):
        """Extract context around keywords"""
        if not isinstance(text, str) or not text.strip():
            return []
            
        text_lower = text.lower()
        tokens = self.custom_word_tokenize(text_lower)
        contexts = []
        
        for i, token in enumerate(tokens):
            # Check if any keywords are in token
            if any(keyword in token for keyword in keywords):
                # Extract context (window of words around the keyword)
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                context = ' '.join(tokens[start:end])
                contexts.append(context)
                
        return contexts
    
    def analyze_sentiment_with_context(self, text):
        """Покращений аналіз сентименту з українською мовою"""
        if not isinstance(text, str) or not text.strip():
            return 0
        
        # Використовуємо менше TextBlob, більше власної логіки
        text_lower = text.lower()
        words = self.custom_word_tokenize(text_lower)
        
        sentiment_score = 0
        negated = False
        intensified = False
        diminished = False
        
        # Трекинг для уникнення подвійного рахунку
        processed_positive_words = set()
        processed_negative_words = set()
        
        for i, word in enumerate(words):
            # Перевірка на заперечення
            if any(neg in word for neg in NEGATIONS):
                negated = True
                continue
            
            # Перевірка на підсилювачі
            if any(intens in word for intens in INTENSIFIERS):
                intensified = True
                diminished = False
                continue
            
            # Перевірка на зменшувачі
            if any(dim in word for dim in DIMINISHERS):
                diminished = True
                intensified = False
                continue
            
            # Підрахунок позитивних слів
            positive_match = next((pos for pos in POSITIVE_WORDS if pos in word and pos not in processed_positive_words), None)
            if positive_match:
                processed_positive_words.add(positive_match)
                modifier = -1 if negated else 1
                
                if intensified:
                    modifier *= 1.8  # Збільшено вплив інтенсифікаторів
                elif diminished:
                    modifier *= 0.4
                    
                sentiment_score += modifier
                
                # Скидання модифікаторів
                negated = False
                intensified = False
                diminished = False
            
            # Підрахунок негативних слів
            negative_match = next((neg for neg in NEGATIVE_WORDS if neg in word and neg not in processed_negative_words), None)
            if negative_match and not positive_match:
                processed_negative_words.add(negative_match)
                modifier = 1 if negated else -1
                
                if intensified:
                    modifier *= 1.8
                elif diminished:
                    modifier *= 0.4
                    
                sentiment_score += modifier
                
                # Скидання модифікаторів
                negated = False
                intensified = False
                diminished = False
            
            # Скидання заперечення після 3 слів
            if negated and i > 0 and (i % 3 == 0):
                negated = False
                
            # Скидання інтенсифікаторів після 2 слів
            if (intensified or diminished) and i > 0 and (i % 2 == 0):
                intensified = False
                diminished = False
        
        # Нормалізація з покращеною формулою
        if sentiment_score != 0:
            normalized_score = sentiment_score / (abs(sentiment_score) + 2)  # Зменшено згладжування
        else:
            normalized_score = 0
        
        # Мінімальний вплив TextBlob тільки для англійських слів
        try:
            blob_sentiment = TextBlob(text).sentiment.polarity * 0.1  # Зменшено вплив
            combined_sentiment = normalized_score + blob_sentiment
        except:
            combined_sentiment = normalized_score
        
        # Обмеження діапазону
        return max(min(combined_sentiment, 1), -1)
    
    def extract_attributes(self, text):
        """Extract tire attributes from text with improved context analysis"""
        attributes_found = {}
        
        if not isinstance(text, str) or not text.strip():
            return attributes_found
            
        text_lower = text.lower()
        sentences = re.split(r'[.!?]', text_lower)
        
        # For each attribute, look for mentions in sentences
        for attribute, keywords in TIRE_ATTRIBUTES.items():
            attribute_mentions = []
            attribute_sentiments = []
            
            for sentence in sentences:
                # Check for presence of attribute keywords in sentence
                if any(keyword in sentence for keyword in keywords):
                    # Extract contexts around keywords
                    contexts = self.extract_context(sentence, keywords, 5)
                    
                    # Analyze sentiment for each context
                    for context in contexts:
                        context_sentiment = self.analyze_sentiment_with_context(context)
                        attribute_mentions.append(context)
                        attribute_sentiments.append(context_sentiment)
            
            # If attribute mentions found
            if attribute_mentions:
                # Calculate average sentiment for attribute
                avg_sentiment = sum(attribute_sentiments) / len(attribute_sentiments)
                
                # Determine attribute weight (number of mentions)
                attribute_weight = len(attribute_mentions)
                
                # Add attribute with its score and weight
                attributes_found[attribute] = {
                    'sentiment': avg_sentiment,
                    'weight': attribute_weight,
                    'mentions': attribute_mentions[:3]  # Save up to 3 example mentions
                }
                
        return attributes_found
    
    def analyze_sentiment(self):
        """Perform sentiment analysis on all reviews"""
        if 'Коментар' not in self.df.columns:
            messagebox.showerror("Помилка", "Колонка 'Коментар' не знайдена в даних")
            return
            
        print("Виконую аналіз сентиментів...")
        
        # Show wait cursor
        self.root.config(cursor="wait")
        self.root.update()
        
        try:
            # Will store sentiment results
            results = []
            
            for _, row in self.df.iterrows():
                review_id = row['ID']
                comment = str(row['Коментар'])
                
                # Overall sentiment
                sentiment = self.analyze_sentiment_with_context(comment)
                
                # Determine sentiment category
                if sentiment > 0.6:
                    sentiment_category = "Дуже позитивний"
                elif sentiment > 0.2:
                    sentiment_category = "Позитивний"
                elif sentiment >= -0.2:
                    sentiment_category = "Нейтральний"
                elif sentiment >= -0.6:
                    sentiment_category = "Негативний"
                else:
                    sentiment_category = "Дуже негативний"
                
                # Extract attributes
                attributes_data = self.extract_attributes(comment)
                
                # Add to results
                review_result = {
                    'ID': review_id,
                    'Коментар': comment,
                    'Загальний_сентимент': sentiment,
                    'Оцінка_сентименту': sentiment_category,
                }
                
                # Add attributes with details
                for attr, attr_data in attributes_data.items():
                    review_result[f'Атрибут_{attr}'] = attr_data['sentiment']
                
                results.append(review_result)
            
            # Create DataFrame with results
            self.analyzed_df = pd.DataFrame(results)
            
            # Add analyzed data back to the original DataFrame
            if len(self.df) == len(self.analyzed_df):
                self.df['Загальний_сентимент'] = self.analyzed_df['Загальний_сентимент']
                self.df['Оцінка_сентименту'] = self.analyzed_df['Оцінка_сентименту']
                
                # Add attribute columns
                attr_columns = [col for col in self.analyzed_df.columns if col.startswith('Атрибут_')]
                for col in attr_columns:
                    self.df[col] = self.analyzed_df[col]
                
                # Update filtered data
                self.filtered_df = self.df.copy()
            
            # Calculate attribute statistics
            self.calculate_sentiment_statistics()
            
            print("Аналіз сентиментів завершено.")
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при аналізі сентиментів: {str(e)}")
            print(f"Помилка при аналізі сентиментів: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Reset cursor
            self.root.config(cursor="")
    
    def calculate_sentiment_statistics(self):
        """Calculate statistics based on sentiment analysis"""
        if self.analyzed_df is None or self.analyzed_df.empty:
            return
            
        # Overall sentiment distribution
        sentiment_counts = self.analyzed_df['Оцінка_сентименту'].value_counts()
        
        # Attributes statistics
        attribute_columns = [col for col in self.analyzed_df.columns if col.startswith('Атрибут_')]
        attributes_details = {}
        
        for col in attribute_columns:
            attr_name = col.replace('Атрибут_', '')
            
            # Filter non-empty values
            valid_values = self.analyzed_df[col].dropna()
            
            if not valid_values.empty:
                # Basic statistics
                attr_mean = valid_values.mean()
                attr_median = valid_values.median()
                attr_std = valid_values.std() if len(valid_values) > 1 else 0
                attr_count = len(valid_values)
                
                # Save detailed information
                attributes_details[attr_name] = {
                    'середня_оцінка': attr_mean,
                    'медіана': attr_median,
                    'стандартне_відхилення': attr_std,
                    'кількість_згадок': attr_count
                }
        
        # Store in sentiment_data for use in UI
        self.sentiment_data = {
            'розподіл_сентиментів': sentiment_counts.to_dict(),
            'атрибути': attributes_details,
            'загальний_сентимент': self.analyzed_df['Загальний_сентимент'].mean()
        }
    
    # 3. DATA HANDLING
    def select_file(self):
        """Open file selection dialog"""
        file_path = filedialog.askopenfilename(
            title="Виберіть файл з відгуками",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx;*.xls"), ("All files", "*.*")]
        )
        
        if file_path:  # User selected a file
            self.file_path = file_path
            self.load_file()
    
    def load_file(self):
        """Завантаження файлу без генерації тестових даних"""
        try:
            if not self.file_path or not os.path.exists(self.file_path):
                messagebox.showerror("Помилка", f"Файл не знайдено: {self.file_path}")
                return
                    
            # Визначення типу файлу
            file_ext = os.path.splitext(self.file_path)[1].lower()
            
            if file_ext == '.csv':
                # Спроба різних кодувань для CSV
                encodings = ['utf-8', 'cp1251', 'iso-8859-1', 'utf-8-sig']
                
                for encoding in encodings:
                    try:
                        self.df = pd.read_csv(self.file_path, encoding=encoding, sep=None, engine='python')
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    messagebox.showerror("Помилка", "Не вдалося розпізнати кодування CSV-файлу")
                    return
            else:  # Excel
                self.df = pd.read_excel(self.file_path)
            
            print(f"Файл завантажено успішно. Кількість рядків: {len(self.df)}")
            print(f"Колонки у файлі: {list(self.df.columns)}")
            
            # Нормалізація колонок
            self.normalize_columns()
            
            # Додання ID якщо відсутній
            if 'ID' not in self.df.columns:
                self.df['ID'] = range(1, len(self.df) + 1)
            
            # Перевірка обов'язкових колонок
            if 'Коментар' not in self.df.columns:
                messagebox.showerror("Помилка", "У файлі відсутня колонка 'Коментар'")
                return
            
            # Створення фільтрованого DataFrame
            self.filtered_df = self.df.copy()
            
            # Аналіз сентиментів
            self.analyze_sentiment()
            
            # Створення фільтрів та оновлення візуалізацій
            self.create_filters()
            self.update_stats()
            self.update_all_visualizations()
            
            messagebox.showinfo("Інформація", f"Файл успішно завантажено: {os.path.basename(self.file_path)}")
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося завантажити файл: {str(e)}")
            print(f"Деталі помилки: {e}")
            import traceback
            traceback.print_exc()

    def create_test_data(self):
        """Create test data for demonstration"""
        messagebox.showinfo("Інформація", "Створюємо тестові дані для демонстрації")
        
        # Create DataFrame with test data
        data = {
            'ID': list(range(1, 51)),
            'Марка_шини': ['Bridgestone', 'Continental', 'Michelin', 'Goodyear', 'Dunlop',
                          'Nokian', 'Pirelli', 'Toyo', 'Yokohama', 'Hankook'] * 5,
            'Модель': ['Model1', 'Model2', 'Model3', 'Model4', 'Model5'] * 10,
            'Рейтинг': [4.5, 3.8, 5.0, 4.2, 3.5, 4.8, 4.0, 3.9, 4.7, 4.1] * 5,
            'Пробіг_км': [50000, 80000, 120000, 60000, 90000, 70000, 85000, 100000, 65000, 110000] * 5,
            'Тип_вантажівки': ['Важка вантажівка', 'Бетономішалка', 'Важка вантажівка', 'Бетономішалка', 'Важка вантажівка'] * 10,
            'Поверхня_доріг': ['Асфальт', 'Гравій', 'Змішана', 'Асфальт', 'Гравій'] * 10,
            'Сезон': ['Зимові', 'Літні', 'Всесезонні', 'Зимові', 'Літні'] * 10,
            'Ціна_грн': [8000, 12000, 15000, 9000, 11000, 14000, 10000, 13000, 8500, 12500] * 5,
            'Коментар': [
                'Шини дуже добре тримають дорогу на мокрому асфальті. Зчеплення відмінне.',
                'Висока ціна, але виправдовує себе відмінною якістю і керованістю.',
                'Чудово справляється у зимових умовах. Рекомендую для важких вантажівок.',
                'Помірна вартість і добра якість. Шини досить тихі при високих швидкостях.',
                'Погано тримається на слизькій дорозі, але в цілому задовільна якість за свою ціну.',
                'Відмінне зчеплення на сухому та вологому асфальті. Добре керується.',
                'Довго витримує без зносу, відмінно працює при високому навантаженні.',
                'Шини стабільні на високих швидкостях, але швидко зношуються на поганих дорогах.',
                'Чудова шина для важких вантажівок, добре тримає дорогу у будь-яких умовах.',
                'Економічна витрата палива, довго служить, відмінно справляється з навантаженням.',
                'Надійні зимові шини, гарно працюють на снігу, але на льоду ковзають.',
                'Дорогі, але виправдовують кожну копійку завдяки надійності та тривалому терміну служби.',
                'Тихі й комфортні шини, але зчеплення у дощ могло би бути кращим.',
                'Повністю задоволений якістю цих шин. Зносостійкість на висоті.',
                'Трохи шумні на швидкості, але гальмівні характеристики чудові.',
                'Не рекомендую для зимових умов, погано тримають сніг і лід.',
                'Ідеальне співвідношення ціни і якості. Зносостійкість просто вражає.',
                'Низький рівень шуму, але на поганих дорогах дещо жорсткі.',
                'Керованість відмінна, але ціна могла б бути нижчою.',
                'Розчарований якістю. Швидко зносились після 30000 км.',
                'Чудовий вибір для далеких рейсів, комфортні і надійні.',
                'Зчеплення із сухим асфальтом відмінне, але на мокрій дорозі не дуже.',
                'Просто жахливі шини! Постійний шум і швидкий знос протектора.',
                'Гарно працюють на усіх типах доріг, рекомендую.',
                'Надійні для перевезення важкого вантажу, стійкі до проколів.',
                'Середні за якістю. Є кращі варіанти за цю ціну.',
                'Відмінно тримають дорогу, не зношуються навіть при повному навантаженні.',
                'Характеристики на льоду і снігу не вражають. Є кращі зимові варіанти.',
                'Добре тримають на слизькій дорозі, але шумні.',
                'Дуже хороші шини, виправдовують всі очікування.',
                'Незадовільна зносостійкість, але гарна керованість.',
                'Комфортні для далеких рейсів, зменшують втому водія.',
                'Відмінні шини для міжміського транспорту. Легкий рух, економія палива.',
                'Не найкращий вибір для бездоріжжя, швидко зношуються.',
                'Задоволений покупкою - тихий хід, гарне зчеплення у будь-яку погоду.',
                'Слабка бокова стінка, уразлива до пошкоджень.',
                'Найкращі шини, які я коли-небудь використовував у моїй вантажівці!',
                'Чудові характеристики, але надто дорогі.',
                'Не виправдали очікувань. Швидко втрачають форму під навантаженням.',
                'Гарні всесезонні шини, підходять для різних дорожніх умов.',
                'Відмінна зносостійкість, але шумні на високих швидкостях.',
                'Не рекомендую для далеких рейсів через підвищений шум.',
                'Хороші шини за свої гроші. Не найкращі, але служать надійно.',
                'Зимові характеристики просто фантастичні! Впевнено тримають дорогу.',
                'Якість виготовлення нижче середньої, протектор швидко зношується.',
                'Ідеальний баланс м`якості й керованості. Дуже задоволений.',
                'Трохи жорсткі на нерівних дорогах, але зносяться дуже повільно.',
                'Низький рівень шуму, плавний хід - просто насолода.',
                'Погано працюють на мокрій дорозі, часто ковзають при гальмуванні.',
                'Оптимальне співвідношення характеристик для далеких перевезень.'
            ]
        }
    
    # 4. FILTER HANDLING
    def create_filters(self):
        """Create filter controls with proper attribute handling"""
        # Clear previous filters
        for widget in self.filter_frame.winfo_children():
            widget.destroy()
                
        self.filter_vars = {}
        self.filter_widgets = {}
        
        # Create button frame for Apply/Reset
        filter_button_frame = ttk.Frame(self.filter_frame)
        filter_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add buttons for applying and resetting filters
        apply_button = ttk.Button(
            filter_button_frame, 
            text="Застосувати фільтри", 
            command=self.apply_filters,
            style='Primary.TButton'
        )
        apply_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        reset_button = ttk.Button(
            filter_button_frame, 
            text="Скинути фільтри", 
            command=self.reset_filters
        )
        reset_button.pack(side=tk.RIGHT, padx=5, fill=tk.X, expand=True)
        
        # Title
        ttk.Label(self.filter_frame, text="Фільтри для аналізу", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Create filters for each column
        filter_columns = [
            'Марка_шини', 'Модель', 'Рейтинг', 'Пробіг_км', 
            'Тип_вантажівки', 'Поверхня_доріг', 'Сезон', 'Ціна_грн', 'Дата_відгуку'
        ]
        
        # Add sentiment filter if available
        if 'Оцінка_сентименту' in self.df.columns:
            filter_columns.append('Оцінка_сентименту')
        
        # Create scrollable frame for filters
        canvas = tk.Canvas(self.filter_frame, borderwidth=0)
        scrollbar = ttk.Scrollbar(self.filter_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")
        
        # Create option for word cloud display mode
        wordcloud_options_frame = ttk.LabelFrame(scrollable_frame, text="Налаштування хмари слів")
        wordcloud_options_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add option to color words based on ratings or sentiment
        sentiment_frame = ttk.Frame(wordcloud_options_frame)
        sentiment_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # ВИПРАВЛЕНІ ЧЕКБОКСИ З ВЗАЄМОВИКЛЮЧНОЮ ЛОГІКОЮ
        # Option for rating-based colors
        color_rating_check = ttk.Checkbutton(
            sentiment_frame, 
            text="Зафарбовувати за оцінками",
            variable=self.color_by_ratings,
            command=self.on_color_rating_changed
        )
        color_rating_check.pack(side=tk.LEFT, padx=5)
        
        # Option for sentiment-based colors
        color_sentiment_check = ttk.Checkbutton(
            sentiment_frame, 
            text="Зафарбовувати за сентиментом",
            variable=self.color_by_sentiment,
            command=self.on_color_sentiment_changed
        )
        color_sentiment_check.pack(side=tk.LEFT, padx=5)
        
        # Add color scheme selection
        color_frame = ttk.Frame(wordcloud_options_frame)
        color_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Label(color_frame, text="Кольорова схема:").pack(side=tk.LEFT)
        
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'Blues', 'Reds', 'Greens', 
                    'Oranges', 'Purples', 'YlOrBr', 'cool', 'hot', 'rainbow', 'tab10']
        
        colormap_combobox = ttk.Combobox(color_frame, textvariable=self.colormap_var, values=colormaps, width=15)
        colormap_combobox.pack(side=tk.LEFT, padx=5)
        colormap_combobox.bind("<<ComboboxSelected>>", lambda e: self.update_wordcloud_settings())
        
        # Add max words controller
        max_words_frame = ttk.Frame(wordcloud_options_frame)
        max_words_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Label(max_words_frame, text="Максимум слів:").pack(side=tk.LEFT)
        
        max_words_scale = ttk.Scale(
            max_words_frame, 
            from_=20, 
            to=200, 
            variable=self.max_words_var,
            orient=tk.HORIZONTAL
        )
        max_words_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        max_words_scale.bind("<ButtonRelease-1>", lambda e: self.update_wordcloud_settings())
        
        max_words_label = ttk.Label(max_words_frame, text=str(self.max_words_var.get()), width=3)
        max_words_label.pack(side=tk.LEFT)
        
        self.max_words_var.trace_add("write", lambda *args: max_words_label.config(text=str(self.max_words_var.get())))
        
        update_wordcloud_btn = ttk.Button(
            wordcloud_options_frame,
            text="Оновити хмару слів",
            command=self.generate_wordcloud
        )
        update_wordcloud_btn.pack(fill=tk.X, padx=5, pady=5)

        # Get all actual attribute columns from the dataframe
        existing_attribute_columns = [col for col in self.df.columns if col.startswith('Атрибут_')]
        
        # Only create attribute filter section if there are potential attributes to filter
        if existing_attribute_columns or len(self.df) > 0:
            # Create attribute filter section with a separate scrollable area
            attributes_filter_frame = ttk.LabelFrame(scrollable_frame, text="Фільтрація за атрибутами шин")
            attributes_filter_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Group attribute filters by category (РОЗШИРЕНИЙ СПИСОК)
            attr_groups = {
                'Зчеплення': ['зчеплення_сухий_асфальт', 'зчеплення_волога_дорога', 'зчеплення_сніг', 'зчеплення_лід', 'волога_дорога_загальне'],
                'Зносостійкість': ['зносостійкість_протектор', 'зносостійкість_загальна', 'стійкість_до_пошкоджень', 'міцність_боковини'],
                'Комфорт': ['шум_рівень', 'шум_характер', 'комфорт_їзда', 'комфорт_вібрації'],
                'Керованість': ['управління_маневреність', 'управління_стабільність', 'управління_точність', 'гальмування'],
                'Сезонність': ['сезонність_літо', 'сезонність_зима', 'сезонність_всесезонність', 'шиповані', 'сніг_глибокий', 'сніг_укатаний', 'лід_ожеледь'],
                'Економічність': ['ціна_якість', 'ціна', 'економічність', 'паливна_ефективність'],
                'Інше': ['протектор_дизайн', 'бездоріжжя', 'поведінка_навантаженні', 'посадка_кріплення', 'якість_виробництва', 'дизайн_зовнішній', 'бренд_довіра', 'країна_виробництва', 'гарантія', 'ціна_доставки']
            }
            
            # Inner scrollable frame for attributes
            attr_canvas = tk.Canvas(attributes_filter_frame, height=150, borderwidth=0)
            attr_scrollbar = ttk.Scrollbar(attributes_filter_frame, orient="vertical", command=attr_canvas.yview)
            attr_scrollable_frame = ttk.Frame(attr_canvas)
            
            attr_scrollable_frame.bind(
                "<Configure>",
                lambda e: attr_canvas.configure(scrollregion=attr_canvas.bbox("all"))
            )
            
            attr_canvas.create_window((0, 0), window=attr_scrollable_frame, anchor="nw")
            attr_canvas.configure(yscrollcommand=attr_scrollbar.set)
            
            attr_canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
            attr_scrollbar.pack(side="right", fill="y")
            
            # Get all existing attribute column names (without the 'Атрибут_' prefix)
            existing_attributes = [col.replace('Атрибут_', '') for col in existing_attribute_columns]
            
            # Process each attribute group
            for group_name, attr_list in attr_groups.items():
                # Check if any attribute in this group exists or can exist
                group_has_attributes = any(attr in existing_attributes for attr in attr_list)
                has_data_to_analyze = len(self.df) > 0
                
                if group_has_attributes or has_data_to_analyze:
                    group_frame = ttk.LabelFrame(attr_scrollable_frame, text=group_name)
                    group_frame.pack(fill=tk.X, padx=5, pady=2)
                    
                    # Process attributes in this group
                    for attr_name in attr_list:
                        attr_col = f"Атрибут_{attr_name}"
                        
                        # Only create UI for attributes that exist or might exist after analysis
                        if attr_col in self.df.columns or has_data_to_analyze:
                            attr_frame = ttk.Frame(group_frame)
                            attr_frame.pack(fill=tk.X, padx=2, pady=2)
                            
                            # Create a checkbox to enable/disable this attribute filter
                            attr_enabled_var = tk.BooleanVar(value=False)
                            attr_check = ttk.Checkbutton(
                                attr_frame, 
                                text=attr_name,
                                variable=attr_enabled_var,
                                state='normal' if attr_col in self.df.columns else 'disabled'
                            )
                            attr_check.pack(side=tk.LEFT)
                            
                            # Create a slider for minimum sentiment value
                            attr_min_var = tk.DoubleVar(value=-1.0)
                            attr_scale = ttk.Scale(
                                attr_frame, 
                                from_=-1.0, 
                                to=1.0, 
                                variable=attr_min_var,
                                orient=tk.HORIZONTAL,
                                state='normal' if attr_col in self.df.columns else 'disabled'
                            )
                            attr_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
                            
                            # Label for the current value
                            attr_label = ttk.Label(attr_frame, text="-1.0", width=5)
                            attr_label.pack(side=tk.LEFT)
                            
                            # Update label when slider changes
                            attr_min_var.trace_add("write", lambda *args, label=attr_label, var=attr_min_var: 
                                label.config(text=f"{var.get():.1f}"))
                            
                            # Store variables in filter_vars for filtering
                            self.filter_vars[attr_col] = {
                                "type": "attribute", 
                                "enabled": attr_enabled_var, 
                                "min_sentiment": attr_min_var,
                                "column": attr_col
                            }
        
        # Create filters for standard columns that exist in the DataFrame
        for column in filter_columns:
            if column in self.df.columns:
                self.create_filter_for_column(column, scrollable_frame)
           
    def create_filter_for_column(self, column, parent_frame):
        # Create frame for filter
        filter_frame = ttk.LabelFrame(parent_frame, text=column)
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Column data type
        dtype = self.df[column].dtype
        
        # Handle different data types
        if pd.api.types.is_numeric_dtype(dtype):
            # Numeric data - create min-max sliders
            min_val = self.df[column].min()
            max_val = self.df[column].max()
            
            # Variables to store values
            min_var = tk.DoubleVar(value=min_val)
            max_var = tk.DoubleVar(value=max_val)
            
            # Labels with current values
            min_label = ttk.Label(filter_frame, text=f"Мін: {min_val:.1f}")
            min_label.pack(anchor=tk.W)
            
            min_scale = ttk.Scale(
                filter_frame, 
                from_=min_val, 
                to=max_val, 
                variable=min_var,
                command=lambda val, lbl=min_label, var=min_var: lbl.config(text=f"Мін: {float(val):.1f}")
            )
            min_scale.pack(fill=tk.X)
            
            max_label = ttk.Label(filter_frame, text=f"Макс: {max_val:.1f}")
            max_label.pack(anchor=tk.W)
            
            max_scale = ttk.Scale(
                filter_frame, 
                from_=min_val, 
                to=max_val, 
                variable=max_var,
                command=lambda val, lbl=max_label, var=max_var: lbl.config(text=f"Макс: {float(val):.1f}")
            )
            max_scale.pack(fill=tk.X)
            
            self.filter_vars[column] = {"type": "range", "min": min_var, "max": max_var}
            self.filter_widgets[column] = {"min_scale": min_scale, "max_scale": max_scale}
            
        elif pd.api.types.is_datetime64_dtype(dtype) or column == 'Дата_відгуку':
                    # Date - create fields for selecting date range
                    try:
                        if column == 'Дата_відгуку' and not pd.api.types.is_datetime64_dtype(dtype):
                            self.df[column] = pd.to_datetime(self.df[column], errors='coerce')
                            
                        min_date = self.df[column].min()
                        max_date = self.df[column].max()
                        
                        # Convert dates to strings for UI display
                        min_date_str = min_date.strftime('%Y-%m-%d') if not pd.isna(min_date) else ""
                        max_date_str = max_date.strftime('%Y-%m-%d') if not pd.isna(max_date) else ""
                        
                        ttk.Label(filter_frame, text="Початкова дата (залиште порожнім для відключення):").pack(anchor=tk.W)
                        start_date_var = tk.StringVar(value="")  # Start with empty value
                        start_date_entry = ttk.Entry(filter_frame, textvariable=start_date_var)
                        start_date_entry.pack(fill=tk.X, pady=2)
                        
                        ttk.Label(filter_frame, text="Кінцева дата (залиште порожнім для відключення):").pack(anchor=tk.W)
                        end_date_var = tk.StringVar(value="")  # Start with empty value
                        end_date_entry = ttk.Entry(filter_frame, textvariable=end_date_var)
                        end_date_entry.pack(fill=tk.X, pady=2)
                        
                        # Add helper text
                        helper_text = f"Доступний діапазон: {min_date_str} - {max_date_str}"
                        ttk.Label(filter_frame, text=helper_text, font=('Arial', 8), foreground='gray').pack(anchor=tk.W)
                        
                        self.filter_vars[column] = {"type": "date", "start": start_date_var, "end": end_date_var}
                        
                    except Exception as e:
                        # If something went wrong with dates, use text filter
                        ttk.Label(filter_frame, text=f"Помилка формату дати: {str(e)}").pack()
                        text_var = tk.StringVar()
                        entry = ttk.Entry(filter_frame, textvariable=text_var)
                        entry.pack(fill=tk.X)
                        self.filter_vars[column] = {"type": "text", "value": text_var}
                
        else:
            # Categorical data - create dropdown or checkboxes
            unique_values = self.df[column].dropna().unique()
            unique_values = sorted(unique_values)
            
            # Special handling for sentiment categories - always use checkboxes
            if column == 'Оцінка_сентименту' or len(unique_values) <= 15:  # For small number of values, use checkboxes
                var_dict = {}
                for val in unique_values:
                    val_var = tk.BooleanVar(value=True)
                    val_frame = ttk.Frame(filter_frame)
                    val_frame.pack(fill=tk.X)
                    
                    check = ttk.Checkbutton(val_frame, text=str(val), variable=val_var)
                    check.pack(side=tk.LEFT)
                    
                    var_dict[val] = val_var
                
                self.filter_vars[column] = {"type": "check", "values": var_dict}
                
            else:  # For many values, use dropdown
                ttk.Label(filter_frame, text="Введіть значення для фільтрації:").pack(anchor=tk.W)
                text_var = tk.StringVar()
                entry = ttk.Entry(filter_frame, textvariable=text_var)
                entry.pack(fill=tk.X, pady=2)
                
                self.filter_vars[column] = {"type": "text", "value": text_var}
                
                # Add dropdown for convenience
                if len(unique_values) <= 100:  # Limit the number for performance
                    values_list = [str(v) for v in unique_values]
                    combo = ttk.Combobox(filter_frame, values=values_list)
                    combo.pack(fill=tk.X, pady=2)
                    
                    # When an item is selected from the list, set the value in the input field
                    combo.bind("<<ComboboxSelected>>", 
                            lambda event, var=text_var: var.set(event.widget.get()))
    
    def apply_filters(self):
        if self.df is None:
            messagebox.showinfo("Інформація", "Спочатку завантажте файл")
            return
            
        # Show the user that processing is in progress
        self.root.config(cursor="wait")
        self.root.update()
            
        # ВАЖЛИВО: Start with a FRESH copy of the full dataset every time
        self.filtered_df = self.df.copy()
        
        print(f"Starting filter application. Original data size: {len(self.df)}, Initial filter data size: {len(self.filtered_df)}")
        
        # Apply filters sequentially
        for filter_key, filter_info in self.filter_vars.items():
            filter_type = filter_info.get("type")
            
            # Handle different filter types
            if filter_type == "range":
                # Skip if column doesn't exist
                if filter_key not in self.df.columns:
                    continue
                    
                min_val = filter_info["min"].get()
                max_val = filter_info["max"].get()
                prev_size = len(self.filtered_df)
                self.filtered_df = self.filtered_df[
                    (self.filtered_df[filter_key] >= min_val) & 
                    (self.filtered_df[filter_key] <= max_val)
                ]
                print(f"Range filter on {filter_key}: {min_val}-{max_val}, rows: {prev_size} -> {len(self.filtered_df)}")
                
            elif filter_type == "date":
                            # Skip if column doesn't exist
                            if filter_key not in self.df.columns:
                                continue
                                
                            try:
                                start_date_str = filter_info["start"].get().strip()
                                end_date_str = filter_info["end"].get().strip()
                                
                                # Only apply date filters if values are provided
                                if start_date_str:
                                    start_date = pd.to_datetime(start_date_str)
                                    prev_size = len(self.filtered_df)
                                    self.filtered_df = self.filtered_df[self.filtered_df[filter_key] >= start_date]
                                    print(f"Date filter (start) on {filter_key}: {start_date_str}, rows: {prev_size} -> {len(self.filtered_df)}")
                                    
                                if end_date_str:
                                    end_date = pd.to_datetime(end_date_str)
                                    prev_size = len(self.filtered_df)
                                    self.filtered_df = self.filtered_df[self.filtered_df[filter_key] <= end_date]
                                    print(f"Date filter (end) on {filter_key}: {end_date_str}, rows: {prev_size} -> {len(self.filtered_df)}")
                                    
                                # If both date fields are empty, don't apply any date filtering
                                if not start_date_str and not end_date_str:
                                    print(f"Date filter on {filter_key}: skipped (no dates specified)")
                                    
                            except Exception as e:
                                print(f"Error applying date filter for {filter_key}: {e}")
                                messagebox.showerror("Помилка дати", f"Неправильний формат дати для {filter_key}: {str(e)}")
                    
            elif filter_type == "check":
                # Skip if column doesn't exist
                if filter_key not in self.df.columns:
                    continue
                    
                # Get list of values for which True is set
                selected_values = [val for val, var in filter_info["values"].items() if var.get()]
                if selected_values:
                    prev_size = len(self.filtered_df)
                    self.filtered_df = self.filtered_df[self.filtered_df[filter_key].isin(selected_values)]
                    print(f"Category filter on {filter_key}: {selected_values}, rows: {prev_size} -> {len(self.filtered_df)}")
                    
            elif filter_type == "text":
                # Skip if column doesn't exist
                if filter_key not in self.df.columns:
                    continue
                    
                text_value = filter_info["value"].get().strip()
                if text_value:
                    # For text fields - case-insensitive search
                    prev_size = len(self.filtered_df)
                    mask = self.filtered_df[filter_key].astype(str).str.contains(text_value, case=False, na=False)
                    self.filtered_df = self.filtered_df[mask]
                    print(f"Text filter on {filter_key}: '{text_value}', rows: {prev_size} -> {len(self.filtered_df)}")
            
            elif filter_type == "attribute":
                # Only apply if filter is enabled
                if filter_info["enabled"].get():
                    column = filter_info["column"]
                    min_sentiment = filter_info["min_sentiment"].get()
                    
                    # Skip if the attribute column doesn't exist
                    if column not in self.df.columns:
                        continue
                        
                    prev_size = len(self.filtered_df)
                    # Filter rows where the attribute is mentioned with sentiment >= min_sentiment
                    mask = self.filtered_df[column].apply(
                        lambda x: not pd.isna(x) and x >= min_sentiment
                    )
                    self.filtered_df = self.filtered_df[mask]
                    attr_name = column.replace('Атрибут_', '')
                    print(f"Attribute filter on {attr_name}: min sentiment={min_sentiment}, rows: {prev_size} -> {len(self.filtered_df)}")
        
        print(f"Final filtered data size: {len(self.filtered_df)}")
        
        # Update statistics first
        self.update_stats()
        
        # Update all visualizations with filtered data
        self.update_all_visualizations()
        
        # Return to normal cursor
        self.root.config(cursor="")
    
    def reset_filters(self):
        if self.df is None:
            return
            
        # Show the user that processing is in progress
        self.root.config(cursor="wait")
        self.root.update()
            
        # ВАЖЛИВО: Restore full dataset з оригінального df
        self.filtered_df = self.df.copy()
        
        print(f"Resetting filters. Original data size: {len(self.df)}, Restored data size: {len(self.filtered_df)}")
        
        # Reset filters to initial values
        for filter_key, filter_info in self.filter_vars.items():
            filter_type = filter_info.get("type")
            
            if filter_type == "range":
                if filter_key in self.df.columns:
                    min_val = self.df[filter_key].min()
                    max_val = self.df[filter_key].max()
                    filter_info["min"].set(min_val)
                    filter_info["max"].set(max_val)
                    
                    # Update labels for scales
                    if filter_key in self.filter_widgets:
                        min_scale = self.filter_widgets[filter_key].get("min_scale")
                        max_scale = self.filter_widgets[filter_key].get("max_scale")
                        
                        if min_scale and min_scale.winfo_exists():
                            min_scale.configure(from_=min_val, to=max_val)
                            
                        if max_scale and max_scale.winfo_exists():
                            max_scale.configure(from_=min_val, to=max_val)
            
            elif filter_type == "date":
                if filter_key in self.df.columns and pd.api.types.is_datetime64_dtype(self.df[filter_key].dtype):
                    min_date = self.df[filter_key].min()
                    max_date = self.df[filter_key].max()
                    
                    min_date_str = min_date.strftime('%Y-%m-%d') if not pd.isna(min_date) else ""
                    max_date_str = max_date.strftime('%Y-%m-%d') if not pd.isna(max_date) else ""
                    
                    filter_info["start"].set(min_date_str)
                    filter_info["end"].set(max_date_str)
            
            elif filter_type == "check":
                # Set True value for all checkboxes
                for var in filter_info["values"].values():
                    var.set(True)
            
            elif filter_type == "text":
                filter_info["value"].set("")
            
            elif filter_type == "attribute":
                # Disable attribute filters and reset min sentiment to -1.0
                filter_info["enabled"].set(False)
                filter_info["min_sentiment"].set(-1.0)
        
        # Force update statistics with original data
        self.update_stats()
        
        # Force update all visualizations with original data
        self.update_all_visualizations()
        
        print(f"Filters reset completed. Current filtered data size: {len(self.filtered_df)}")
        
        # Return to normal cursor
        self.root.config(cursor="")
    
    # 5. UI CREATION
    def create_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top panel with file load button and export button
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)
        
        title_label = ttk.Label(top_frame, text="Аналіз відгуків про вантажні шини", font=("Arial", 16, "bold"))
        title_label.pack(side=tk.LEFT, padx=5)
        
        # Style for button
        style = ttk.Style()
        style.configure('Big.TButton', font=('Arial', 11))
        
        # Export button
        export_button = ttk.Button(
            top_frame, 
            text="Експортувати результати", 
            command=self.export_results,
            style='Big.TButton'
        )
        export_button.pack(side=tk.RIGHT, padx=5, ipadx=10, ipady=5)
        
        # Load button with improved appearance
        load_button = ttk.Button(
            top_frame, 
            text="Завантажити файл", 
            command=self.select_file,
            style='Big.TButton'
        )
        load_button.pack(side=tk.RIGHT, padx=5, ipadx=10, ipady=5)
        
        # Main container
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Measure the width of the filter buttons to determine minimum width
        temp_button = ttk.Button(self.root, text="Застосувати фільтри")
        temp_button.pack()
        button_width = temp_button.winfo_reqwidth() * 2 + 20  # Width of two buttons plus padding
        temp_button.destroy()

        # Create paned window for flexible separation of filter area and visualizations
        paned_window = ttk.PanedWindow(content_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)

        # Filter panel (left)
        self.filter_frame = ttk.LabelFrame(paned_window, text="Фільтри")
        self.filter_frame.configure(width=button_width)  # Set minimum width
        paned_window.add(self.filter_frame, weight=1)
        
        # Panel for visualizations (right)
        viz_frame = ttk.Frame(paned_window)
        paned_window.add(viz_frame, weight=3)
        
        # Panel for displaying word cloud and charts
        self.notebook = ttk.Notebook(viz_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab for word cloud
        self.wordcloud_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.wordcloud_frame, text="Хмара слів")
        
        # Tab for ratings chart
        self.ratings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ratings_frame, text="Аналіз оцінок")
        
        # Tab for brand comparison
        self.brands_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.brands_frame, text="Порівняння брендів")
        
        # Tab for sentiment analysis
        self.sentiment_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sentiment_frame, text="Аналіз сентименту")
        
        # Tab for attribute analysis
        self.attributes_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.attributes_frame, text="Аналіз атрибутів")
        
        # Create figure for word cloud
        self.fig_wordcloud = Figure(figsize=(10, 8), dpi=100, tight_layout=True)
        self.ax_wordcloud = self.fig_wordcloud.add_subplot(111)
        self.ax_wordcloud.axis('off')
        self.ax_wordcloud.set_title('Завантажте файл для аналізу', fontsize=20, fontweight='bold', pad=20)
        
        # Place chart in interface
        self.canvas_wordcloud = FigureCanvasTkAgg(self.fig_wordcloud, master=self.wordcloud_frame)
        self.canvas_wordcloud_widget = self.canvas_wordcloud.get_tk_widget()
        self.canvas_wordcloud_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add toolbar for word cloud chart
        toolbar_frame_wordcloud = ttk.Frame(self.wordcloud_frame)
        toolbar_frame_wordcloud.pack(fill=tk.X)
        self.toolbar_wordcloud = NavigationToolbar2Tk(self.canvas_wordcloud, toolbar_frame_wordcloud)
        self.toolbar_wordcloud.update()
        
        # Create figure for ratings chart
        self.fig_ratings = Figure(figsize=(10, 8), dpi=100, tight_layout=True)
        self.ax_ratings = self.fig_ratings.add_subplot(111)
        self.ax_ratings.set_title('Аналіз оцінок', fontsize=20, fontweight='bold', pad=20)
        
        # Place ratings chart in interface
        self.canvas_ratings = FigureCanvasTkAgg(self.fig_ratings, master=self.ratings_frame)
        self.canvas_ratings_widget = self.canvas_ratings.get_tk_widget()
        self.canvas_ratings_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add toolbar for ratings chart
        toolbar_frame_ratings = ttk.Frame(self.ratings_frame)
        toolbar_frame_ratings.pack(fill=tk.X)
        self.toolbar_ratings = NavigationToolbar2Tk(self.canvas_ratings, toolbar_frame_ratings)
        self.toolbar_ratings.update()
        
        # Create figure for brand comparison
        self.fig_brands = Figure(figsize=(10, 8), dpi=100, tight_layout=True)
        self.ax_brands = self.fig_brands.add_subplot(111)
        self.ax_brands.set_title('Порівняння брендів', fontsize=20, fontweight='bold', pad=20)
        
        # Place brands chart in interface
        self.canvas_brands = FigureCanvasTkAgg(self.fig_brands, master=self.brands_frame)
        self.canvas_brands_widget = self.canvas_brands.get_tk_widget()
        self.canvas_brands_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add toolbar for brands chart
        toolbar_frame_brands = ttk.Frame(self.brands_frame)
        toolbar_frame_brands.pack(fill=tk.X)
        self.toolbar_brands = NavigationToolbar2Tk(self.canvas_brands, toolbar_frame_brands)
        self.toolbar_brands.update()
        
        # Create figure for sentiment analysis
        self.fig_sentiment = Figure(figsize=(10, 8), dpi=100, tight_layout=True)
        self.ax_sentiment = self.fig_sentiment.add_subplot(111)
        self.ax_sentiment.set_title('Аналіз сентименту відгуків', fontsize=20, fontweight='bold', pad=20)
        
        # Place sentiment chart in interface
        self.canvas_sentiment = FigureCanvasTkAgg(self.fig_sentiment, master=self.sentiment_frame)
        self.canvas_sentiment_widget = self.canvas_sentiment.get_tk_widget()
        self.canvas_sentiment_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add toolbar for sentiment chart
        toolbar_frame_sentiment = ttk.Frame(self.sentiment_frame)
        toolbar_frame_sentiment.pack(fill=tk.X)
        self.toolbar_sentiment = NavigationToolbar2Tk(self.canvas_sentiment, toolbar_frame_sentiment)
        self.toolbar_sentiment.update()
        
        # Create figure for attributes analysis
        self.fig_attributes = Figure(figsize=(10, 8), dpi=100, tight_layout=True)
        self.ax_attributes = self.fig_attributes.add_subplot(111)
        self.ax_attributes.set_title('Аналіз атрибутів шин', fontsize=20, fontweight='bold', pad=20)
        
        # Place attributes chart in interface
        self.canvas_attributes = FigureCanvasTkAgg(self.fig_attributes, master=self.attributes_frame)
        self.canvas_attributes_widget = self.canvas_attributes.get_tk_widget()
        self.canvas_attributes_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add toolbar for attributes chart
        toolbar_frame_attributes = ttk.Frame(self.attributes_frame)
        toolbar_frame_attributes.pack(fill=tk.X)
        self.toolbar_attributes = NavigationToolbar2Tk(self.canvas_attributes, toolbar_frame_attributes)
        self.toolbar_attributes.update()
        
        # Bottom panel with statistics
        self.stats_frame = ttk.LabelFrame(main_frame, text="Статистика")
        self.stats_frame.pack(fill=tk.X, pady=5)
        
        # Split statistical information into sub-panels for better organization
        stats_subframe = ttk.Frame(self.stats_frame)
        stats_subframe.pack(fill=tk.X, padx=5, pady=5)
        
        # Left statistics panel (basic information)
        left_stats_frame = ttk.Frame(stats_subframe)
        left_stats_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.basic_stats_label = ttk.Label(left_stats_frame, text="Завантажте файл для відображення статистики", font=("Arial", 9))
        self.basic_stats_label.pack(anchor=tk.W)
        
        # Right statistics panel (additional information)
        right_stats_frame = ttk.Frame(stats_subframe)
        right_stats_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        self.extra_stats_label = ttk.Label(right_stats_frame, text="", font=("Arial", 9))
        self.extra_stats_label.pack(anchor=tk.E)
        
        # Sentiment statistics panel
        sentiment_stats_frame = ttk.Frame(self.stats_frame)
        sentiment_stats_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.sentiment_stats_label = ttk.Label(sentiment_stats_frame, text="", font=("Arial", 9))
        self.sentiment_stats_label.pack(anchor=tk.W)
    
    # 6. STATISTICS AND VISUALIZATION
    def update_stats(self):
        """Update statistical information"""
        if self.filtered_df is None:
            return
        
        # ВАЖЛИВО: Переконуємось що у нас є правильні дані
        if self.df is None:
            return
            
        # Basic statistics
        total_records = len(self.df)
        filtered_records = len(self.filtered_df)
        
        print(f"Stats update: total_records = {total_records}, filtered_records = {filtered_records}")

        # Rating statistics
        if 'Рейтинг' in self.filtered_df.columns:
            avg_rating = self.filtered_df['Рейтинг'].mean()
            min_rating = self.filtered_df['Рейтинг'].min()
            max_rating = self.filtered_df['Рейтинг'].max()
        
            # Format basic statistics
            basic_stats = f"Записів: {filtered_records} з {total_records} | "
            basic_stats += f"Середня оцінка: {avg_rating:.2f} | "
            basic_stats += f"Діапазон оцінок: {min_rating:.1f}-{max_rating:.1f}"
        
            self.basic_stats_label.config(text=basic_stats)
        else:
            # If there's no ratings column
            basic_stats = f"Записів: {filtered_records} з {total_records}"
            self.basic_stats_label.config(text=basic_stats)
        
        # Additional statistics
        extra_stats = ""
        
        # If there's a brand column, show the most popular ones
        if 'Марка_шини' in self.filtered_df.columns and len(self.filtered_df) > 0:
            top_brands = self.filtered_df['Марка_шини'].value_counts().head(3)
            if not top_brands.empty:
                brands_str = ', '.join([f"{brand} ({count})" for brand, count in top_brands.items()])
                extra_stats += f"Топ-3 бренди: {brands_str}"
            
        self.extra_stats_label.config(text=extra_stats)
        
        # Update sentiment statistics
        sentiment_stats = ""
        if 'Загальний_сентимент' in self.filtered_df.columns and len(self.filtered_df) > 0:
            avg_sentiment = self.filtered_df['Загальний_сентимент'].mean()
            
            # Determine sentiment category
            if avg_sentiment > 0.6:
                sentiment_category = "дуже позитивне"
            elif avg_sentiment > 0.2:
                sentiment_category = "позитивне"
            elif avg_sentiment >= -0.2:
                sentiment_category = "нейтральне"
            elif avg_sentiment >= -0.6:
                sentiment_category = "негативне"
            else:
                sentiment_category = "дуже негативне"
                
            sentiment_stats = f"Середній сентимент: {avg_sentiment:.2f} ({sentiment_category})"
            
            # Add sentiment distribution
            if 'Оцінка_сентименту' in self.filtered_df.columns:
                sentiment_counts = self.filtered_df['Оцінка_сентименту'].value_counts()
                if not sentiment_counts.empty:
                    top_sentiment = sentiment_counts.idxmax()
                    top_count = sentiment_counts.max()
                    sentiment_stats += f" | Домінуючий сентимент: {top_sentiment} ({top_count} відгуків)"
        
        self.sentiment_stats_label.config(text=sentiment_stats)
    
    def update_wordcloud_settings(self):
        """Оновлення налаштувань хмари слів з взаємовиключними опціями"""
        # Перевіряємо, яка кнопка була натиснута останньою
        if self.color_by_ratings.get() and self.color_by_sentiment.get():
            # Якщо обидві вибрані, вимикаємо сентимент (пріоритет оцінкам)
            self.color_by_sentiment.set(False)
        
        self.generate_wordcloud()
    
    def on_color_rating_changed(self):
        """Обробник зміни чекбокса оцінок"""
        if self.color_by_ratings.get():
            self.color_by_sentiment.set(False)
        self.generate_wordcloud()

    def on_color_sentiment_changed(self):
        """Обробник зміни чекбокса сентименту"""
        if self.color_by_sentiment.get():
            self.color_by_ratings.set(False)
        self.generate_wordcloud()
    
    def update_all_visualizations(self):
        # Show cursor to indicate processing
        self.root.config(cursor="wait")
        self.root.update()

        try:
            # Generate all visualizations in the proper order
            self.generate_wordcloud()
            self.generate_ratings_chart()
            self.generate_brands_chart()
            self.generate_sentiment_charts()  # This calls both sentiment and attribute charts
            
            # Redraw all canvases
            self.canvas_wordcloud.draw()
            self.canvas_ratings.draw()
            self.canvas_brands.draw()
            self.canvas_sentiment.draw()
            self.canvas_attributes.draw()
            
            # Update the notebook tab to show a badge if there's a filter active
            if len(self.filtered_df) < len(self.df):
                for i in range(self.notebook.index("end")):
                    tab_id = self.notebook.tabs()[i]
                    current_text = self.notebook.tab(tab_id, "text")
                    
                    # Add filter indicator if not already present
                    if not current_text.endswith(" 🔍"):
                        self.notebook.tab(tab_id, text=f"{current_text} 🔍")
            else:
                # Remove filter indicators if all data is shown
                for i in range(self.notebook.index("end")):
                    tab_id = self.notebook.tabs()[i]
                    current_text = self.notebook.tab(tab_id, "text")
                    
                    # Remove filter indicator if present
                    if current_text.endswith(" 🔍"):
                        self.notebook.tab(tab_id, text=current_text.replace(" 🔍", ""))
            
        except Exception as e:
            messagebox.showerror("Помилка", f"Помилка при оновленні візуалізацій: {str(e)}")
            print(f"Помилка при оновленні візуалізацій: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Return to normal cursor
            self.root.config(cursor="")
            
        # Update statistics as well
        self.update_stats()

    def generate_wordcloud(self):
        if self.filtered_df is None or len(self.filtered_df) == 0:
            self.ax_wordcloud.clear()
            self.ax_wordcloud.text(0.5, 0.5, 'Немає даних для аналізу. Перевірте фільтри.',
                horizontalalignment='center', verticalalignment='center',
                fontsize=16, fontweight='bold')
            self.canvas_wordcloud.draw()
            return

        try:
            # First, check for the comment column
            if 'Коментар' not in self.filtered_df.columns:
                self.ax_wordcloud.clear()
                self.ax_wordcloud.text(0.5, 0.5, 'У даних відсутня колонка "Коментар"',
                                      horizontalalignment='center', verticalalignment='center',
                                      fontsize=16, fontweight='bold')
                self.canvas_wordcloud.draw()
                return

            # Define text cleaning function locally
            def clean_text(text):
                if not isinstance(text, str):
                    return ""
                # Convert to lowercase
                text = text.lower()
                # Remove numbers
                text = re.sub(r'\d+', '', text)
                # Remove special characters and punctuation
                text = re.sub(r'[^\w\s]', ' ', text)
                # Replace multiple spaces with a single one
                text = re.sub(r'\s+', ' ', text)
                return text.strip()

            # Copy data for analysis
            df_analysis = self.filtered_df.copy().reset_index(drop=True)
            
            # Check for duplicate columns and fix if needed
            if df_analysis.columns.duplicated().any():
                # Get duplicate column names
                dupes = df_analysis.columns[df_analysis.columns.duplicated()].tolist()
                
                # If 'Коментар' is duplicated, merge the content
                if 'Коментар' in dupes:
                    # Get all columns named 'Коментар'
                    comment_cols = [col for col in df_analysis.columns if col == 'Коментар']
                    
                    # Create a new column with merged comments
                    df_analysis['Combined_Comments'] = ""
                    for idx, row in df_analysis.iterrows():
                        all_comments = []
                        for col in comment_cols:
                            try:
                                if isinstance(row[col], str) and row[col].strip():
                                    all_comments.append(row[col])
                            except:
                                pass
                        
                        df_analysis.at[idx, 'Combined_Comments'] = " ".join(all_comments)
                    
                    # Drop the original 'Коментар' columns and rename the combined one
                    for col in comment_cols:
                        df_analysis = df_analysis.drop(col, axis=1)
                    
                    df_analysis = df_analysis.rename(columns={'Combined_Comments': 'Коментар'})

            # Check for ratings and sentiment
            has_ratings = 'Рейтинг' in df_analysis.columns
            has_sentiment = 'Загальний_сентимент' in df_analysis.columns

            # Process comments to extract words
            all_words = []
            word_rating_dict = {}
            word_sentiment_dict = {}
            word_count_dict = {}
            
            # Process each comment
            for idx, row in df_analysis.iterrows():
                try:
                    # Get comment text safely
                    comment_val = row['Коментар']
                    
                    # Handle different types of comment_val
                    if isinstance(comment_val, pd.Series):
                        # If it's a Series, extract text that might be a comment
                        # Look for labels that might contain comments
                        possible_comment_indexes = ['Коментар', 'відгук', 'comment', 'text']
                        comment_text = ""
                        
                        for index in possible_comment_indexes:
                            if index in comment_val.index:
                                comment_text += " " + str(comment_val[index])
                        
                        # If no matching index found, use the whole Series as a string
                        if not comment_text.strip():
                            comment_text = str(comment_val)
                            
                    elif isinstance(comment_val, pd.DataFrame):
                        # If it's a DataFrame, try to extract the comment column
                        if 'Коментар' in comment_val.columns:
                            comment_text = " ".join(comment_val['Коментар'].astype(str))
                        else:
                            # Just convert to string as a fallback
                            comment_text = str(comment_val)
                    else:
                        # For other types (string, etc.), convert to string
                        comment_text = str(comment_val)
                    
                    # Clean the comment text
                    cleaned_text = clean_text(comment_text)
                    
                    # Extract words
                    words = cleaned_text.split()
                    
                    # Filter and count words
                    for word in words:
                        if word not in self.all_stopwords and len(word) > 2:
                            all_words.append(word)
                            
                            # Track word ratings if available
                            if has_ratings and self.color_by_ratings.get():
                                try:
                                    rating = float(row['Рейтинг'])
                                    if not pd.isna(rating):
                                        rating_min = df_analysis['Рейтинг'].min()
                                        rating_max = df_analysis['Рейтинг'].max()
                                        norm_rating = (rating - rating_min) / (rating_max - rating_min) if rating_max > rating_min else 0.5
                                        
                                        if word in word_rating_dict:
                                            word_rating_dict[word] += norm_rating
                                            word_count_dict[word] += 1
                                        else:
                                            word_rating_dict[word] = norm_rating
                                            word_count_dict[word] = 1
                                except:
                                    # Skip if rating conversion fails
                                    pass
                            
                            # Track word sentiment if available
                            if has_sentiment and self.color_by_sentiment.get():
                                try:
                                    sentiment = float(row['Загальний_сентимент'])
                                    if not pd.isna(sentiment):
                                        # Normalize sentiment to 0-1 range (from -1 to 1)
                                        norm_sentiment = (sentiment + 1) / 2
                                        
                                        if word in word_sentiment_dict:
                                            word_sentiment_dict[word] += norm_sentiment
                                            word_count_dict[word] = word_count_dict.get(word, 0) + 1
                                        else:
                                            word_sentiment_dict[word] = norm_sentiment
                                            word_count_dict[word] = 1
                                except:
                                    # Skip if sentiment conversion fails
                                    pass
                            
                except Exception as e:
                    # Silently continue if there's an error with a comment
                    continue
            
            # Create word frequency counter
            self.word_freq = Counter(all_words)

            # If no words found, use default Ukrainian text
            if not self.word_freq:
                default_text = """
                шини зчеплення дорога надійні гарні якісні міцні вантажівка зима тихі
                керованість стійкість довговічність економія паливо асфальт дощ сніг
                ціна термін служби витривалість протектор вологий гальмування безпека
                маневреність комфорт вібрація
                """
                # Remove stopwords from default text
                default_words = []
                for word in default_text.split():
                    word = word.strip()
                    if word and word not in self.all_stopwords and len(word) > 2:
                        default_words.append(word)
                
                self.word_freq = Counter(default_words)
                
                # Notify the user we're using demo data
                messagebox.showinfo("Інформація", 
                    "Недостатньо даних для створення повноцінної хмари слів. Використовуємо демонстраційні дані.")

            # Generate the wordcloud with appropriate coloring
            if has_ratings and self.color_by_ratings.get() and word_rating_dict:
                # Calculate average rating per word
                avg_word_rating = {
                    word: total_rating / word_count_dict[word]
                    for word, total_rating in word_rating_dict.items()
                }

                # Generate color function based on ratings
                def color_func(word, **kwargs):
                    if word in avg_word_rating:
                        rating_score = avg_word_rating[word]
                        # Color from red (0) to green (1)
                        r = int(255 * (1 - rating_score))
                        g = int(255 * rating_score)
                        b = 0
                        return f"rgb({r}, {g}, {b})"
                    return "rgb(128, 128, 128)"  # Default gray

                # Generate word cloud with colors based on ratings
                wordcloud = WordCloud(
                    width=800, 
                    height=600,
                    background_color='white',
                    max_words=self.max_words_var.get(),
                    max_font_size=200,
                    random_state=42,
                    color_func=color_func,
                    collocations=False,
                    stopwords=self.all_stopwords
                )
                
            elif has_sentiment and self.color_by_sentiment.get() and word_sentiment_dict:
                # Calculate average sentiment per word
                avg_word_sentiment = {
                    word: total_sentiment / word_count_dict[word]
                    for word, total_sentiment in word_sentiment_dict.items()
                }

                # Generate color function based on sentiment
                def color_func(word, **kwargs):
                    if word in avg_word_sentiment:
                        sentiment_score = avg_word_sentiment[word]
                        # Color from red (0) to green (1)
                        r = int(255 * (1 - sentiment_score))
                        g = int(255 * sentiment_score)
                        b = 0
                        return f"rgb({r}, {g}, {b})"
                    return "rgb(128, 128, 128)"  # Default gray

                # Generate word cloud with colors based on sentiment
                wordcloud = WordCloud(
                    width=800, 
                    height=600,
                    background_color='white',
                    max_words=self.max_words_var.get(),
                    max_font_size=200,
                    random_state=42,
                    color_func=color_func,
                    collocations=False,
                    stopwords=self.all_stopwords
                )
            else:
                # Generate word cloud with standard colormap
                wordcloud = WordCloud(
                    width=800, 
                    height=600,
                    background_color='white',
                    max_words=self.max_words_var.get(),
                    max_font_size=200,
                    random_state=42,
                    colormap=self.colormap_var.get(),
                    collocations=False,
                    stopwords=self.all_stopwords
                )

            # Generate word cloud from frequency dictionary
            wordcloud.generate_from_frequencies(self.word_freq)
            self.wordcloud = wordcloud
            
            # Clear previous plot
            self.ax_wordcloud.clear()
            
            # Display word cloud
            self.ax_wordcloud.imshow(wordcloud, interpolation='bilinear')
            self.ax_wordcloud.axis('off')
            
            # Set title
            if len(self.filtered_df) < len(self.df):
                title = f'Хмара слів (фільтр: {len(self.filtered_df)} з {len(self.df)} відгуків)'
            else:
                title = 'Хмара слів для всіх відгуків'
                
            # Add information about coloring if used
            if has_ratings and self.color_by_ratings.get():
                title += '\nКольори за оцінками: червоний (низькі) → зелений (високі)'
            elif has_sentiment and self.color_by_sentiment.get():
                title += '\nКольори за сентиментом: червоний (негативний) → зелений (позитивний)'
                
            self.ax_wordcloud.set_title(title, fontsize=16, fontweight='bold', pad=20)

        except Exception as e:
            self.ax_wordcloud.clear()
            self.ax_wordcloud.text(0.5, 0.5, f'Помилка при створенні хмари слів: {str(e)}',
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='red')
            print(f"Помилка при створенні хмари слів: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def generate_ratings_chart(self):
        """Generate ratings analysis chart"""
        if self.filtered_df is None or len(self.filtered_df) == 0:
            self.ax_ratings.clear()
            self.ax_ratings.text(0.5, 0.5, 'Немає даних для аналізу. Перевірте фільтри.',
                horizontalalignment='center', verticalalignment='center',
                fontsize=16, fontweight='bold')
            self.canvas_ratings.draw()
            return
            
        try:
            # Check for ratings column
            if 'Рейтинг' not in self.filtered_df.columns:
                self.ax_ratings.clear()
                self.ax_ratings.text(0.5, 0.5, 'Немає даних про оцінки',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=16, fontweight='bold')
                self.canvas_ratings.draw()
                return
                
            # Clear previous chart
            self.ax_ratings.clear()
            
            # Create ratings histogram
            ratings = self.filtered_df['Рейтинг'].dropna()
            
            if len(ratings) == 0:
                self.ax_ratings.text(0.5, 0.5, 'Немає даних про оцінки',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=16, fontweight='bold')
                self.canvas_ratings.draw()
                return
                
            # Configure histogram step based on data type
            if ratings.dtype in [np.int64, np.int32, np.int16]:
                # For integer ratings
                bins = range(int(min(ratings)), int(max(ratings))+2)
                self.ax_ratings.hist(ratings, bins=bins, alpha=0.7, color='#4CAF50')
                self.ax_ratings.set_xticks(range(int(min(ratings)), int(max(ratings))+1))
            else:
                # For fractional ratings
                bins = np.arange(float(min(ratings)), float(max(ratings))+0.5, 0.5)
                self.ax_ratings.hist(ratings, bins=bins, alpha=0.7, color='#4CAF50')
                
            # Add vertical line for average value
            avg_rating = ratings.mean()
            self.ax_ratings.axvline(avg_rating, color='red', linestyle='dashed', linewidth=2, 
                                label=f'Середня оцінка: {avg_rating:.2f}')
            
            # Chart configuration
            self.ax_ratings.set_title('Розподіл оцінок', fontsize=16, fontweight='bold')
            self.ax_ratings.set_xlabel('Оцінка', fontsize=12)
            self.ax_ratings.set_ylabel('Кількість відгуків', fontsize=12)
            self.ax_ratings.grid(axis='y', linestyle='--', alpha=0.7)
            self.ax_ratings.legend()
            
            # Add text with quantitative information
            info_text = f'Загальна кількість оцінок: {len(ratings)}\n'
            info_text += f'Середня оцінка: {avg_rating:.2f}\n'
            info_text += f'Медіана: {ratings.median():.2f}\n'
            info_text += f'Мін. оцінка: {ratings.min():.1f}\n'
            info_text += f'Макс. оцінка: {ratings.max():.1f}'
            
            # Add text block
            self.ax_ratings.text(0.95, 0.95, info_text,
                horizontalalignment='right', verticalalignment='top',
                transform=self.ax_ratings.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
            
            # If data is filtered, add a note
            if len(self.filtered_df) < len(self.df):
                filter_text = f'Показано {len(self.filtered_df)} з {len(self.df)} відгуків (застосовані фільтри)'
                self.ax_ratings.text(0.5, 0.01, filter_text,
                    horizontalalignment='center', verticalalignment='bottom',
                    transform=self.ax_ratings.transAxes, fontsize=10, 
                    bbox=dict(facecolor='yellow', alpha=0.2, boxstyle='round,pad=0.3'))
            
            # Update chart
            self.fig_ratings.tight_layout()
            
        except Exception as e:
            self.ax_ratings.clear()
            self.ax_ratings.text(0.5, 0.5, f'Помилка при створенні графіка: {str(e)}',
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='red')
            print(f"Помилка при створенні графіка оцінок: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def generate_brands_chart(self):
        if self.filtered_df is None or len(self.filtered_df) == 0:
            self.ax_brands.clear()
            self.ax_brands.text(0.5, 0.5, 'Немає даних для аналізу. Перевірте фільтри.',
                horizontalalignment='center', verticalalignment='center',
                fontsize=16, fontweight='bold')
            self.canvas_brands.draw()
            return
            
        try:
            # Check for brand and rating columns
            if 'Марка_шини' not in self.filtered_df.columns:
                self.ax_brands.clear()
                self.ax_brands.text(0.5, 0.5, 'Немає даних про бренди',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=16, fontweight='bold')
                self.canvas_brands.draw()
                return
                
            # Clear previous chart
            self.ax_brands.clear()
            
            # Group data by brands
            if 'Рейтинг' in self.filtered_df.columns and len(self.filtered_df) > 0:
                # If there are ratings, calculate average rating for each brand
                brand_stats = self.filtered_df.groupby('Марка_шини')['Рейтинг'].agg(['mean', 'count'])
                
                # Ensure index is string to avoid duplicate labels
                brand_stats.index = [str(x) for x in brand_stats.index]
                
                # Sort by count and get top brands
                top_n = min(10, len(brand_stats))
                if top_n > 0:
                    brand_data = brand_stats.nlargest(top_n, 'count')
                    
                    # Sort by mean rating for display
                    brand_data = brand_data.sort_values(by='mean', ascending=False)
                    
                    # Get x-labels and data
                    brands = brand_data.index.tolist()
                    means = brand_data['mean'].tolist()
                    counts = brand_data['count'].tolist()
                    
                    # Create bar chart with numeric x-axis
                    x_pos = np.arange(len(brands))
                    bars = self.ax_brands.bar(x_pos, means, alpha=0.7, color='#4CAF50')
                    
                    # Add labels with review counts
                    for i, (count, mean) in enumerate(zip(counts, means)):
                        self.ax_brands.text(i, mean+0.1, f"n={count}", 
                                        ha='center', va='bottom', fontsize=9)
                    
                    # Chart configuration
                    self.ax_brands.set_title('Середні оцінки за брендами', fontsize=16, fontweight='bold')
                    self.ax_brands.set_xlabel('Бренд', fontsize=12)
                    self.ax_brands.set_ylabel('Середня оцінка', fontsize=12)
                    self.ax_brands.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Add horizontal line for average rating across all brands
                    avg_rating = self.filtered_df['Рейтинг'].mean()
                    self.ax_brands.axhline(avg_rating, color='red', linestyle='dashed', linewidth=2, 
                                    label=f'Загальна середня: {avg_rating:.2f}')
                    self.ax_brands.legend()
                    
                    # Set the x-tick positions and labels
                    self.ax_brands.set_xticks(x_pos)
                    self.ax_brands.set_xticklabels(brands, rotation=45, ha='right')
                    
                    # Set y-axis limits to a reasonable range for ratings (0 to 5 or max if higher)
                    max_y = max(5.5, max(means) + 0.5)
                    self.ax_brands.set_ylim([0, max_y])
                else:
                    self.ax_brands.text(0.5, 0.5, 'Недостатньо даних для порівняння брендів',
                                    horizontalalignment='center', verticalalignment='center',
                                    fontsize=14)
            else:
                # If there are no ratings or no data, just count reviews for each brand
                if len(self.filtered_df) > 0:
                    brand_counts = self.filtered_df['Марка_шини'].value_counts()
                    
                    # Ensure index is string to avoid duplicate labels
                    brand_counts.index = [str(x) for x in brand_counts.index]
                    
                    # Select top-N brands
                    top_n = min(10, len(brand_counts))
                    if top_n > 0:
                        brand_counts = brand_counts.nlargest(top_n)
                        
                        # Sort by count
                        brand_counts = brand_counts.sort_values(ascending=False)
                        
                        # Get x-labels and data
                        brands = brand_counts.index.tolist()
                        counts = brand_counts.values.tolist()
                        
                        # Create bar chart with numeric x-axis
                        x_pos = np.arange(len(brands))
                        bars = self.ax_brands.bar(x_pos, counts, alpha=0.7, color='#2196F3')
                        
                        # Add labels with counts
                        for i, v in enumerate(counts):
                            self.ax_brands.text(i, v+0.5, str(v), ha='center', va='bottom')
                        
                        # Chart configuration
                        self.ax_brands.set_title('Кількість відгуків за брендами', fontsize=16, fontweight='bold')
                        self.ax_brands.set_xlabel('Бренд', fontsize=12)
                        self.ax_brands.set_ylabel('Кількість відгуків', fontsize=12)
                        self.ax_brands.grid(axis='y', linestyle='--', alpha=0.7)
                        
                        # Set the x-tick positions and labels
                        self.ax_brands.set_xticks(x_pos)
                        self.ax_brands.set_xticklabels(brands, rotation=45, ha='right')
                    else:
                        self.ax_brands.text(0.5, 0.5, 'Недостатньо даних для порівняння брендів',
                                        horizontalalignment='center', verticalalignment='center',
                                        fontsize=14)
                else:
                    self.ax_brands.text(0.5, 0.5, 'Немає даних для аналізу',
                                    horizontalalignment='center', verticalalignment='center',
                                    fontsize=14)
            
            # If data is filtered, add a note
            if len(self.filtered_df) < len(self.df):
                filter_text = f'Показано {len(self.filtered_df)} з {len(self.df)} відгуків (застосовані фільтри)'
                self.ax_brands.text(0.5, 0.01, filter_text,
                    horizontalalignment='center', verticalalignment='bottom',
                    transform=self.ax_brands.transAxes, fontsize=10, 
                    bbox=dict(facecolor='yellow', alpha=0.2, boxstyle='round,pad=0.3'))
            
            # Update chart
            self.fig_brands.tight_layout()
            self.canvas_brands.draw()
            
        except Exception as e:
            self.ax_brands.clear()
            self.ax_brands.text(0.5, 0.5, f'Помилка при створенні графіка: {str(e)}',
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='red')
            print(f"Помилка при створенні графіка брендів: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def generate_sentiment_charts(self):
        """Generate sentiment analysis charts"""
        # First, check if we have sentiment data
        if self.filtered_df is None or len(self.filtered_df) == 0 or 'Загальний_сентимент' not in self.filtered_df.columns:
            self.ax_sentiment.clear()
            self.ax_sentiment.text(0.5, 0.5, 'Немає даних про сентимент. Перевірте фільтри.',
                horizontalalignment='center', verticalalignment='center',
                fontsize=16, fontweight='bold')
            self.canvas_sentiment.draw()
            
            self.ax_attributes.clear()
            self.ax_attributes.text(0.5, 0.5, 'Немає даних про атрибути. Перевірте фільтри.',
                horizontalalignment='center', verticalalignment='center',
                fontsize=16, fontweight='bold')
            self.canvas_attributes.draw()
            return
            
        try:
            # Generate main sentiment distribution chart
            self.generate_sentiment_distribution()
            
            # Generate attributes chart
            self.generate_attributes_chart()
            
        except Exception as e:
            self.ax_sentiment.clear()
            self.ax_sentiment.text(0.5, 0.5, f'Помилка при створенні графіка сентименту: {str(e)}',
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, color='red')
            print(f"Помилка при створенні графіка сентименту: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def generate_sentiment_distribution(self):
        """Generate sentiment distribution chart"""
        # Clear previous chart
        self.ax_sentiment.clear()
        
        # Get sentiment data
        sentiments = self.filtered_df['Загальний_сентимент'].dropna()
        
        if len(sentiments) == 0:
            self.ax_sentiment.text(0.5, 0.5, 'Немає даних про сентимент',
                horizontalalignment='center', verticalalignment='center',
                fontsize=16, fontweight='bold')
            self.canvas_sentiment.draw()
            return
            
        # Create histogram
        bins = np.linspace(-1, 1, 21)  # 20 bins from -1 to 1
        
        # Get histogram counts
        n, bins, patches = self.ax_sentiment.hist(sentiments, bins=bins, alpha=0.7)
        
        # Color bars based on sentiment value
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        for i, patch in enumerate(patches):
            # Calculate color: red at -1, yellow at 0, green at 1
            if bin_centers[i] < 0:
                # Red to yellow for negative
                r = 1.0
                g = 1.0 + bin_centers[i]
                b = 0.0
            else:
                # Yellow to green for positive
                r = 1.0 - bin_centers[i]
                g = 1.0
                b = 0.0
            patch.set_facecolor((r, g, b))
        
        # Add vertical line for average sentiment
        avg_sentiment = sentiments.mean()
        self.ax_sentiment.axvline(avg_sentiment, color='black', linestyle='dashed', linewidth=2, 
                            label=f'Середній сентимент: {avg_sentiment:.2f}')
        
        # Add sentiment categories as vertical spans with labels
        span_props = dict(alpha=0.1, lw=0)
        
        # Very negative region: -1.0 to -0.6
        self.ax_sentiment.axvspan(-1.0, -0.6, facecolor='red', **span_props)
        self.ax_sentiment.text(-0.8, self.ax_sentiment.get_ylim()[1]*0.9, 'Дуже\nнегативний', 
                       ha='center', fontsize=8)
        
        # Negative region: -0.6 to -0.2
        self.ax_sentiment.axvspan(-0.6, -0.2, facecolor='orange', **span_props)
        self.ax_sentiment.text(-0.4, self.ax_sentiment.get_ylim()[1]*0.9, 'Негативний', 
                       ha='center', fontsize=8)
        
        # Neutral region: -0.2 to 0.2
        self.ax_sentiment.axvspan(-0.2, 0.2, facecolor='yellow', **span_props)
        self.ax_sentiment.text(0, self.ax_sentiment.get_ylim()[1]*0.9, 'Нейтральний', 
                       ha='center', fontsize=8)
        
        # Positive region: 0.2 to 0.6
        self.ax_sentiment.axvspan(0.2, 0.6, facecolor='yellowgreen', **span_props)
        self.ax_sentiment.text(0.4, self.ax_sentiment.get_ylim()[1]*0.9, 'Позитивний', 
                       ha='center', fontsize=8)
        
        # Very positive region: 0.6 to 1.0
        self.ax_sentiment.axvspan(0.6, 1.0, facecolor='green', **span_props)
        self.ax_sentiment.text(0.8, self.ax_sentiment.get_ylim()[1]*0.9, 'Дуже\nпозитивний', 
                       ha='center', fontsize=8)
        
        # Chart configuration
        self.ax_sentiment.set_title('Розподіл сентименту', fontsize=16, fontweight='bold')
        self.ax_sentiment.set_xlabel('Оцінка сентименту (-1: дуже негативний, +1: дуже позитивний)', fontsize=12)
        self.ax_sentiment.set_ylabel('Кількість відгуків', fontsize=12)
        self.ax_sentiment.grid(axis='y', linestyle='--', alpha=0.7)
        self.ax_sentiment.legend()
        
        # Add sentiment category counts if available
        if 'Оцінка_сентименту' in self.filtered_df.columns:
            sentiment_counts = self.filtered_df['Оцінка_сентименту'].value_counts()
            
            info_text = 'Розподіл за категоріями:\n'
            for category, count in sentiment_counts.items():
                percent = count / len(self.filtered_df) * 100
                info_text += f"{category}: {count} ({percent:.1f}%)\n"
                
            # Add text block
            self.ax_sentiment.text(0.95, 0.95, info_text,
                horizontalalignment='right', verticalalignment='top',
                transform=self.ax_sentiment.transAxes,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                fontsize=10)
        
        # If data is filtered, add a note
        if len(self.filtered_df) < len(self.df):
            filter_text = f'Показано {len(self.filtered_df)} з {len(self.df)} відгуків (застосовані фільтри)'
            self.ax_sentiment.text(0.5, 0.01, filter_text,
                horizontalalignment='center', verticalalignment='bottom',
                transform=self.ax_sentiment.transAxes, fontsize=10, 
                bbox=dict(facecolor='yellow', alpha=0.2, boxstyle='round,pad=0.3'))
        
        # Update chart
        self.fig_sentiment.tight_layout()
        self.canvas_sentiment.draw()
    
    def generate_attributes_chart(self):
        """Generate attribute analysis chart"""
        # Clear previous chart
        self.ax_attributes.clear()
        
        # Get attribute columns
        attribute_columns = [col for col in self.filtered_df.columns if col.startswith('Атрибут_')]
        
        if not attribute_columns:
            self.ax_attributes.text(0.5, 0.5, 'Немає даних про атрибути шин',
                horizontalalignment='center', verticalalignment='center',
                fontsize=16, fontweight='bold')
            self.canvas_attributes.draw()
            return
            
        # Prepare data for bar chart
        attributes = []
        sentiment_scores = []
        mention_counts = []
        
        for col in attribute_columns:
            # Get attribute name without prefix
            attr_name = col.replace('Атрибут_', '')
            
            # Get non-null values for this attribute
            values = self.filtered_df[col].dropna()
            
            if len(values) > 0:
                avg_sentiment = values.mean()
                attributes.append(attr_name)
                sentiment_scores.append(avg_sentiment)
                mention_counts.append(len(values))
        
        if not attributes:
            self.ax_attributes.text(0.5, 0.5, 'Недостатньо даних для аналізу атрибутів',
                horizontalalignment='center', verticalalignment='center',
                fontsize=16, fontweight='bold')
            self.canvas_attributes.draw()
            return
            
        # Sort by count (highest first)
        sorted_indices = np.argsort(mention_counts)[::-1]
        
        # Limit to top 15 attributes to make chart readable
        if len(sorted_indices) > 15:
            sorted_indices = sorted_indices[:15]
            
        # Reorder data
        attributes = [attributes[i] for i in sorted_indices]
        sentiment_scores = [sentiment_scores[i] for i in sorted_indices]
        mention_counts = [mention_counts[i] for i in sorted_indices]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(attributes))
        
        # Create bar with a color corresponding to sentiment
        bars = self.ax_attributes.barh(y_pos, mention_counts, align='center', alpha=0.7)
        
        # Color bars based on sentiment
        for i, bar in enumerate(bars):
            score = sentiment_scores[i]
            if score < -0.6:
                bar.set_color('red')
            elif score < -0.2:
                bar.set_color('orangered')
            elif score < 0.2:
                bar.set_color('gold')
            elif score < 0.6:
                bar.set_color('yellowgreen')
            else:
                bar.set_color('green')
                
            # Add sentiment value as text
            self.ax_attributes.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                             f"{sentiment_scores[i]:.2f}", 
                             va='center', ha='left', fontsize=9)
        
        # Set labels and title
        self.ax_attributes.set_yticks(y_pos)
        self.ax_attributes.set_yticklabels(attributes)
        self.ax_attributes.invert_yaxis()  # Labels read top-to-bottom
        self.ax_attributes.set_xlabel('Кількість згадувань', fontsize=12)
        self.ax_attributes.set_title('Згадування атрибутів шин та їх сентимент', fontsize=16, fontweight='bold')
        
        # Add a color legend for sentiment ranges
        import matplotlib.patches as mpatches
        
        legend_elements = [
            mpatches.Patch(facecolor='red', label='Дуже негативний (-1.0 до -0.6)'),
            mpatches.Patch(facecolor='orangered', label='Негативний (-0.6 до -0.2)'),
            mpatches.Patch(facecolor='gold', label='Нейтральний (-0.2 до 0.2)'),
            mpatches.Patch(facecolor='yellowgreen', label='Позитивний (0.2 до 0.6)'),
            mpatches.Patch(facecolor='green', label='Дуже позитивний (0.6 до 1.0)')
        ]
        
        self.ax_attributes.legend(handles=legend_elements, loc='lower right', fontsize=9)
        
        # If data is filtered, add a note
        if len(self.filtered_df) < len(self.df):
            filter_text = f'Показано {len(self.filtered_df)} з {len(self.df)} відгуків (застосовані фільтри)'
            self.ax_attributes.text(0.5, 0.01, filter_text,
                horizontalalignment='center', verticalalignment='bottom',
                transform=self.ax_attributes.transAxes, fontsize=10, 
                bbox=dict(facecolor='yellow', alpha=0.2, boxstyle='round,pad=0.3'))
        
        # Update chart
        self.fig_attributes.tight_layout()
        self.canvas_attributes.draw()
    
    # 7. EXPORT AND REPORTING
    def export_results(self):
        """Export analysis results to Excel file with enhanced tire attribute data"""
        if self.filtered_df is None or len(self.filtered_df) == 0:
            messagebox.showinfo("Інформація", "Немає даних для експорту")
            return
            
        try:
            output_file = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                initialfile="tire_analysis_with_sentiment.xlsx"
            )
            
            if output_file:
                # Create Excel file with multiple sheets
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    # Save filtered data
                    self.filtered_df.to_excel(writer, sheet_name='Відфільтровані дані', index=False)
                    
                    # Save brand statistics
                    if 'Марка_шини' in self.filtered_df.columns:
                        # Create combined brand statistics with ratings and sentiment if available
                        brand_stats_data = {}
                        
                        if 'Рейтинг' in self.filtered_df.columns:
                            brand_rating_stats = self.filtered_df.groupby('Марка_шини')['Рейтинг'].agg(['count', 'mean', 'min', 'max']).reset_index()
                            brand_rating_stats.columns = ['Бренд', 'Кількість відгуків', 'Середня оцінка', 'Мінімальна оцінка', 'Максимальна оцінка']
                            brand_stats_data.update({col: brand_rating_stats[col] for col in brand_rating_stats.columns})
                        else:
                            brand_counts = self.filtered_df['Марка_шини'].value_counts().reset_index()
                            brand_counts.columns = ['Бренд', 'Кількість відгуків']
                            brand_stats_data.update({col: brand_counts[col] for col in brand_counts.columns})
                        
                        # Add sentiment data if available
                        if 'Загальний_сентимент' in self.filtered_df.columns:
                            brand_sentiment_stats = self.filtered_df.groupby('Марка_шини')['Загальний_сентимент'].agg(['mean', 'min', 'max']).reset_index()
                            brand_sentiment_stats.columns = ['Бренд', 'Середній сентимент', 'Мінімальний сентимент', 'Максимальний сентимент']
                            
                            # Make sure we don't duplicate the brand column
                            for col in brand_sentiment_stats.columns:
                                if col != 'Бренд':
                                    brand_stats_data[col] = brand_sentiment_stats[col]
                        
                        # Create the combined DataFrame and save it
                        brand_stats_df = pd.DataFrame(brand_stats_data)
                        brand_stats_df.to_excel(writer, sheet_name='Статистика по брендах', index=False)
                    
                    # Save sentiment statistics if available
                    if 'Загальний_сентимент' in self.filtered_df.columns:
                        # Overall sentiment statistics
                        sentiment_stats = pd.DataFrame({
                            'Метрика': ['Середній сентимент', 'Медіана', 'Мінімум', 'Максимум', 'Стандартне відхилення'],
                            'Значення': [
                                self.filtered_df['Загальний_сентимент'].mean(),
                                self.filtered_df['Загальний_сентимент'].median(),
                                self.filtered_df['Загальний_сентимент'].min(),
                                self.filtered_df['Загальний_сентимент'].max(),
                                self.filtered_df['Загальний_сентимент'].std()
                            ]
                        })
                        sentiment_stats.to_excel(writer, sheet_name='Статистика сентименту', index=False)
                        
                        # Save distribution of sentiment categories if available
                        if 'Оцінка_сентименту' in self.filtered_df.columns:
                            sentiment_categories = self.filtered_df['Оцінка_сентименту'].value_counts().reset_index()
                            sentiment_categories.columns = ['Категорія', 'Кількість відгуків']
                            sentiment_categories.to_excel(writer, sheet_name='Категорії сентименту', index=False)
                    
                    # Save attribute statistics if available
                    attribute_columns = [col for col in self.filtered_df.columns if col.startswith('Атрибут_')]
                    if attribute_columns:
                        attribute_stats_data = []
                        
                        for col in attribute_columns:
                            attr_name = col.replace('Атрибут_', '')
                            values = self.filtered_df[col].dropna()
                            
                            if len(values) > 0:
                                attribute_stats_data.append({
                                    'Атрибут': attr_name,
                                    'Кількість згадувань': len(values),
                                    'Середній сентимент': values.mean(),
                                    'Мінімальний сентимент': values.min(),
                                    'Максимальний сентимент': values.max(),
                                    'Стандартне відхилення': values.std() if len(values) > 1 else 0
                                })
                        
                        if attribute_stats_data:
                            attribute_stats_df = pd.DataFrame(attribute_stats_data)
                            attribute_stats_df.sort_values('Кількість згадувань', ascending=False, inplace=True)
                            attribute_stats_df.to_excel(writer, sheet_name='Статистика атрибутів', index=False)
                    
                    # Save attribute details for each brand if available
                    if 'Марка_шини' in self.filtered_df.columns and attribute_columns:
                        # Create a detailed brand-attribute matrix with sentiment values
                        brands = self.filtered_df['Марка_шини'].unique()
                        brand_attribute_data = []
                        
                        for brand in brands:
                            brand_df = self.filtered_df[self.filtered_df['Марка_шини'] == brand]
                            
                            # Create a row for each brand
                            brand_row = {'Бренд': brand, 'Кількість відгуків': len(brand_df)}
                            
                            # Add sentiment data for each attribute
                            for col in attribute_columns:
                                attr_name = col.replace('Атрибут_', '')
                                values = brand_df[col].dropna()
                                
                                if len(values) > 0:
                                    brand_row[f'{attr_name} (згадувань)'] = len(values)
                                    brand_row[f'{attr_name} (сентимент)'] = values.mean()
                                else:
                                    brand_row[f'{attr_name} (згадувань)'] = 0
                                    brand_row[f'{attr_name} (сентимент)'] = None
                            
                            brand_attribute_data.append(brand_row)
                        
                        if brand_attribute_data:
                            brand_attribute_df = pd.DataFrame(brand_attribute_data)
                            brand_attribute_df.to_excel(writer, sheet_name='Бренди+Атрибути', index=False)
                    
                    # Save most frequent words
                    if hasattr(self, 'word_freq') and self.word_freq:
                        # Convert to DataFrame
                        word_freq_df = pd.DataFrame(self.word_freq.most_common(100), columns=['Слово', 'Частота'])
                        word_freq_df.to_excel(writer, sheet_name='Частота слів', index=False)
                    
                    # Save attribute examples with mentions
                    if attribute_columns:
                        # Extract examples for each attribute
                        attribute_examples = []
                        
                        # Process each review in the filtered data
                        for idx, row in self.filtered_df.iterrows():
                            comment = row['Коментар']
                            
                            # Skip if comment is not a string
                            if not isinstance(comment, str):
                                continue
                                
                            # For each attribute
                            for attr, keywords in TIRE_ATTRIBUTES.items():
                                # Check if this attribute is mentioned in the comment
                                if any(keyword in comment.lower() for keyword in keywords):
                                    # Find the sentiment for this attribute
                                    attr_sentiment = None
                                    attr_col = f'Атрибут_{attr}'
                                    if attr_col in self.filtered_df.columns:
                                        attr_sentiment = row.get(attr_col)
                                    
                                    # Add to examples
                                    attribute_examples.append({
                                        'Атрибут': attr,
                                        'Коментар': comment,
                                        'Бренд': row.get('Марка_шини', 'Невідомо'),
                                        'Рейтинг': row.get('Рейтинг', 'Невідомо'),
                                        'Сентимент атрибуту': attr_sentiment,
                                        'Загальний сентимент': row.get('Загальний_сентимент', 'Невідомо')
                                    })
                        
                        if attribute_examples:
                            examples_df = pd.DataFrame(attribute_examples)
                            examples_df.to_excel(writer, sheet_name='Приклади атрибутів', index=False)
                    
                messagebox.showinfo("Успіх", f"Результати аналізу збережено: {output_file}")
                
        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося експортувати результати: {str(e)}")
            print(f"Помилка при експорті результатів: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 8. EVENT HANDLERS
    def on_tab_changed(self, event):
        """Handle tab change events to ensure visualizations are up to date"""
        # Get the currently selected tab
        current_tab = self.notebook.index(self.notebook.select())
        
        # Configure cursor to indicate loading
        self.root.config(cursor="wait")
        self.root.update()
        
        try:
            # Refresh the visualization for the selected tab
            if current_tab == 0:  # Word cloud tab
                self.generate_wordcloud()
                self.canvas_wordcloud.draw()
            elif current_tab == 1:  # Ratings tab
                self.generate_ratings_chart()
                self.canvas_ratings.draw()
            elif current_tab == 2:  # Brands tab
                self.generate_brands_chart()
                self.canvas_brands.draw()
            elif current_tab == 3:  # Sentiment tab
                self.generate_sentiment_charts()
            elif current_tab == 4:  # Attributes tab
                self.generate_attributes_chart()
        except Exception as e:
            print(f"Error updating tab {current_tab}: {str(e)}")
        finally:
            # Reset cursor
            self.root.config(cursor="")

    
    def setup_event_handlers(self):
        """Set up all event handlers"""
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # Add export button to the top panel
        export_button = ttk.Button(
            self.root, 
            text="Експортувати результати", 
            command=self.export_results,
            style='Big.TButton'
        )
        export_button.pack(side=tk.TOP, padx=10, pady=10)

# Application start
if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedTireAnalysisApp(root)
    root.mainloop()
