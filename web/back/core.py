import os
import json
import numpy as np
import tensorflow as tf 
import pandas as pd
from tensorflow.keras.models import load_model

# Switch for production and develop mode
ON_PRODUCTION = False
MODEL_DEV = "./model/base_model_dev.keras"
MODEL_PRO = "./model/base_model.keras"
DATA_PATH = "./data/data.csv"

def get_default_model():
    '''
    Return the current model dynamically based on the development stage
    '''
    model_current = MODEL_PRO if ON_PRODUCTION else MODEL_DEV
    bas_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(bas_dir, model_current)
    return model_path

def get_data_file(file=DATA_PATH):
    '''
    Return the path of the data file (data.csv)
    '''
    bas_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(bas_dir, file)
    return data_path

class Model:
    def __init__(self, model_path=None):
        self.model_path = self.__validate_path(model_path)
        self.model = load_model(self.model_path)
        self.data_path = get_data_file()
        self.column_defaults = [
            tf.constant([0.0], dtype=tf.float32) for _ in range(15)]
        self.num_inx = [0, 1, 11, 12]
        self.cat_inx = [2, 3, 4, 5, 6, 7, 8, 9, 10, 13]
        self.num_mean = np.array([2.82031589, 2.63612889, 78.73644655, 33.20106438])
        self.num_std = np.array([1.17156264, 1.19396015, 10.67027785, 9.64125973])
        self.n_cat = [3, 9, 7, 2, 2, 2, 3, 2, 10, 11]

    def predict_one(self, student):
        if not student:
            raise ValueError("Invalid student data provided.")
        # Validate the student object
        student.first_year_persistence = 0
        validation_errors = student.validate()
        if validation_errors:
            raise ValueError(f"Student validation failed: {', '.join(validation_errors)}")
        try:
            line = student.to_line()
            prediction = self.__predict_line(line)
            return prediction.numpy()
        except Exception as e:
            raise RuntimeError(f"An error occurred during prediction: {str(e)}")        

    def train_one(self, student_labelled):
        if not student_labelled:
            raise ValueError("Invalid student data provided.")
        if student.first_year_persistence == 'UKN':
            raise ValueError("Label must be provided.")
        validation_errors = student.validate()
        if validation_errors:
            raise ValueError(f"Student validation failed: {', '.join(validation_errors)}")
        try:
            line = student_labelled.to_line()
            self.__train_line(line)
            self.__save()
            float_line = student_labelled.to_float_line()
            self.__write_line(float_line)
        except Exception as e:
            raise RuntimeError(f"An error occurred during train: {str(e)}") 

    def train_batch(self, new_csv_path):
        try:
            csv_path = get_data_file(new_csv_path)
            dataset = self.__process_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to process the file at {new_csv_path}: {str(e)}")
        try:
            history = self.model.fit(dataset, epochs=1, verbose=0)
            self.__save()
            self.__write_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to train the model: {str(e)}")
        return history.history['accuracy'], history.history['loss']

    def __process_csv(self, file_paths, n_readers=5, n_read_threads=tf.data.AUTOTUNE,
                        shuffle_buffer_size=1500, batch_size=32):
        dataset = tf.data.Dataset.list_files(file_paths)
        dataset = dataset.interleave(   # use interleave to read data
            lambda filepath: tf.data.TextLineDataset(filepath).skip(1), # skip header
            cycle_length=n_readers, # num of elements to read parallelly
            num_parallel_calls=n_read_threads)  # num of threads
        # shuffle, preprocess, batch, and prefetch
        dataset = dataset.shuffle(shuffle_buffer_size) \
            .map(self.__process_line, num_parallel_calls=n_read_threads) \
            .batch(batch_size) \
            .prefetch(tf.data.AUTOTUNE)
        return dataset

    @tf.function
    def __predict_line(self, line):
        features, _ = self.__process_line(line)
        features = tf.reshape(features, (1, -1))
        probabilities = self.model(features)
        prediction = probabilities[0][1]
        return prediction
    
    @tf.function
    def __train_line(self, line):
        x, y = self.__process_line(line)
        x = tf.reshape(x, (1, -1))
        y = tf.reshape(y, (1, -1))
        self.model.fit(x, y, epochs=1, verbose=0)

    @tf.function
    def __process_line(self, line):
        fields = tf.io.decode_csv(line, record_defaults=self.column_defaults)
        # normalize numerical data
        for i in range(len(self.num_inx)):
            col_inx = self.num_inx[i]
            mean = self.num_mean[i]
            std = self.num_std[i]
            fields[col_inx] = (fields[col_inx] - mean) / std
        # convert categorical data
        for i in range(len(self.cat_inx)):
            col_inx = self.cat_inx[i]
            fields[col_inx] = tf.cast(fields[col_inx], tf.int32)
            fields[col_inx] = tf.one_hot(fields[col_inx], depth=self.n_cat[i])
        # target encoding
        fields[-1] = tf.cast(fields[-1], tf.int32)
        fields[-1] = tf.one_hot(fields[-1], depth=2)
        # build x, y
        x = [tf.reshape(f, [-1]) if f.shape.ndims == 0 else f for f in fields[:-1]]
        x = tf.concat([v for v in x], axis=0)
        y = fields[-1]
        return x, y
    
    def __save(self):
        self.model.save(self.model_path)

    def __write_line(self, float_line):
        df_line = pd.DataFrame([float_line])
        df_line.to_csv(self.data_path, mode='a', index=False, header=False)

    def __write_csv(self, new_csv_path):
        df_csv = pd.read_csv(new_csv_path)
        df_csv.to_csv(self.data_path, mode='a', index=False, header=False)

    def __validate_path(self, model_path):
        if model_path is None or model_path not in [MODEL_DEV, MODEL_PRO]:
            default_path = get_default_model()
            return default_path
        else:
            return model_path
    
class Student:
    # Get location of the metadata file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Directory of the script
    MAP_PATH = os.path.join(BASE_DIR, "../config/metadata.json")

    # Load mappings during class initialization
    with open(MAP_PATH, "r") as file:
        metadata = json.load(file)
        value_mappings = {
                key: {int(k): v for k, v in mapping.items()}
                for key, mapping in metadata["VALUE_MAPPINGS"].items()
            }
        numeric_ranges = metadata["NUMERIC_RANGES"]

    def __init__(self, first_term_gpa, second_term_gpa, first_language, funding, school, fast_track, 
                 coop, residency, gender, previous_education, age_group, high_school_average_mark, 
                 math_score, english_grade, first_year_persistence='UKN'):
        # Attributes
        self.first_term_gpa = first_term_gpa
        self.second_term_gpa = second_term_gpa
        self.first_language = first_language
        self.funding = funding
        self.school = school
        self.fast_track = fast_track
        self.coop = coop
        self.residency = residency
        self.gender = gender
        self.previous_education = previous_education
        self.age_group = age_group
        self.high_school_average_mark = high_school_average_mark
        self.math_score = math_score
        self.english_grade = english_grade
        self.first_year_persistence = first_year_persistence

    def to_line(self):
        """
        Converts the student object to a CSV string for input pipelines.
        """
        return ",".join(map(str, [
            self.first_term_gpa,
            self.second_term_gpa,
            self.first_language,
            self.funding,
            self.school,
            self.fast_track,
            self.coop,
            self.residency,
            self.gender,
            self.previous_education,
            self.age_group,
            self.high_school_average_mark,
            self.math_score,
            self.english_grade,
            self.first_year_persistence
        ]))

    def to_float_line(self):
        """
        Converts the student object to a CSV string for input pipelines.
        """
        return [
            self.first_term_gpa,
            self.second_term_gpa,
            self.first_language,
            self.funding,
            self.school,
            self.fast_track,
            self.coop,
            self.residency,
            self.gender,
            self.previous_education,
            self.age_group,
            self.high_school_average_mark,
            self.math_score,
            self.english_grade,
            self.first_year_persistence
        ]

    def validate(self):
        """
        Validates the student attributes against the provided mappings and numeric ranges.
        """
        errors = []
        # Validate categorical values
        for field, mappings in Student.value_mappings.items():
            value = getattr(self, field, None)
            if value not in mappings.keys():
                errors.append(f"{field} value '{value}' is not valid.")
        # Validate numeric ranges
        for field, (min_val, max_val) in Student.numeric_ranges.items():
            value = getattr(self, field, None)
            if not (min_val <= value <= max_val):
                errors.append(f"{field} value '{value}' is out of range [{min_val}, {max_val}].")
        # Validate label if existed.
        if self.first_year_persistence not in [0, 1, 'UKN']:
            errors.append(f"First_year_persistence can only be 0 or 1 but got {self.first_year_persistence}.")

        return errors

    @staticmethod
    def json_to_student(data):
        """
        Creates a Student instance from a JSON dictionary.
        """
        student = Student(
            first_term_gpa=float(data.get("firstTermGpa", 0.0)),
            second_term_gpa=float(data.get("secondTermGpa", 0.0)),
            first_language=float(data.get("firstLanguage", 0.0)),
            funding=float(data.get("funding", 0.0)),
            school=float(data.get("school", 0.0)),
            fast_track=float(data.get("fastTrack", 0.0)),
            coop=float(data.get("coop", 0.0)),
            residency=float(data.get("residency", 0.0)),
            gender=float(data.get("gender", 0.0)),
            previous_education=float(data.get("previousEducation", 0.0)),
            age_group=float(data.get("ageGroup", 0.0)),
            high_school_average_mark=float(data.get("highSchoolAverageMark", 0.0)),
            math_score=float(data.get("mathScore", 0.0)),
            english_grade=float(data.get("englishGrade", 0.0)),
        )
        errors = student.validate()
        if errors:
            raise ValueError(f"Validation errors: {errors}")
        return student

    
if __name__ == "__main__": 
    # student = Student(2.125,2.136364,1.0,2.0,6.0,2.0,1.0,1.0,2.0,1.0,2.0,73.0,18.0,7.0,1.0)
    # model = Model()
    # model.train_one(student)

    # student = Student(0.6,1.4,2,8,3,2,1,2,3,1,8,2.5,4,8)
    # model = Model()
    # prediction = model.predict_one(student)
    # print(prediction)

    csv_file = './data/data_uploaded.csv'
    model = Model()
    acc, loss = model.train_batch(csv_file)
    print(acc, loss)

    

