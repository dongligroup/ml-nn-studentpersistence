# data_map.py

# File path for the raw data
DATA_RAW_PATH = "./data/raw.csv"
DATA_PATH = "./data/data.csv"

# Row need to skip when reading raw data
SKIP_ROWS = 24

# Metadata about the dataset
METADATA = {
    "independent_variables": [
        "First Term Gpa",
        "Second Term Gpa",
        "First Language",
        "Funding",
        "School",
        "Fast Track",
        "Coop",
        "Residency",
        "Gender",
        "Previous Education",
        "Age Group",
        "High School Average Mark",
        "Math Score",
        "English Grade",
    ],
    "dependent_variable": "FirstYearPersistence",
}

# Value mappings for categorical variables
VALUE_MAPPINGS = {
    "First Language": {1: "English", 2: "French", 3: "Other"},
    "Funding": {
        1: "Apprentice_PS",
        2: "GPOG_FT",
        3: "Intl Offshore",
        4: "Intl Regular",
        5: "Intl Transfer",
        6: "Joint Program Ryerson",
        7: "Joint Program UTSC",
        8: "Second Career Program",
        9: "Work Safety Insurance Board",
    },
    "School": {
        1: "Advancement",
        2: "Business",
        3: "Communications",
        4: "Community and Health",
        5: "Hospitality",
        6: "Engineering",
        7: "Transportation",
    },
    "Fast Track": {1: "Yes", 2: "No"},
    "Coop": {1: "Yes", 2: "No"},
    "Residency": {1: "Domestic", 2: "International"},
    "Gender": {1: "Female", 2: "Male", 3: "Neutral"},
    "Previous Education": {1: "HighSchool", 2: "PostSecondary"},
    "Age Group": {
        1: "0 to 18",
        2: "19 to 20",
        3: "21 to 25",
        4: "26 to 30",
        5: "31 to 35",
        6: "36 to 40",
        7: "41 to 50",
        8: "51 to 60",
        9: "61 to 65",
        10: "66+",
    },
    "English Grade": {
        1: "Level-130",
        2: "Level-131",
        3: "Level-140",
        4: "Level-141",
        5: "Level-150",
        6: "Level-151",
        7: "Level-160",
        8: "Level-161",
        9: "Level-170",
        10: "Level-171",
        11: "Level-180",
    },
}

# Numeric ranges for continuous variables
NUMERIC_RANGES = {
    "First Term Gpa": (0.0, 4.5),
    "Second Term Gpa": (0.0, 4.5),
    "High School Average Mark": (0.0, 100.0),
    "Math Score": (0.0, 50.0),
}
