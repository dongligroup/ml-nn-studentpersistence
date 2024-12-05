import React, { useState } from 'react'; 
import axios from 'axios';
import './PredictionForm.css';
import cutegradImage from '../assets/cutegrad.png'; 

        // main form - react form 
const PredictionForm = () => {
    const [studentData, setStudentData] = useState({
        firstTermGpa: '', 
        secondTermGpa: '',
        firstLanguage: '',
        funding: '',
        school: '',
        fastTrack: '',
        coop: '',
        residency: '',
        gender: '',
        previousEducation: '',
        ageGroup: '',
        highSchoolAverageMark: '',
        mathScore: '',
        englishGrade: ''
    });

    const [predictionResult, setPredictionResult] = useState(null);
    const [error, setError] = useState('');
    const [validationError, setValidationError] = useState('');
    const [loading, setLoading] = useState(false);
    

    const handleChange = (e) => {
        const { name, value } = e.target;
        setStudentData({
            ...studentData,
            [name]: value,
        });
    };

    const validateForm = () => {
        const requiredFields = [
            'firstTermGpa', 'secondTermGpa', 'firstLanguage', 'funding',
            'school', 'fastTrack', 'coop', 'residency', 'gender',
            'previousEducation', 'ageGroup', 'highSchoolAverageMark',
            'mathScore', 'englishGrade'
        ];

        for (let field of requiredFields) {
            if (!studentData[field]) {
                return `Field "${field}" is required.`;
            }
        }

        if (isNaN(studentData.firstTermGpa) || isNaN(studentData.secondTermGpa)) {
            return 'First Term GPA and Second Term GPA must be valid numbers.';
        }

        return null; // No errors
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        // validate form
        const validationMessage = validateForm();
        if (validationMessage) {
            setValidationError(validationMessage);
            return;
        }

        setValidationError(''); // clear validation errors
        setLoading(true); // 

        try {
            const response = await axios.post('http://localhost:5001/predict', {
                features: {
                    firstTermGpa: parseFloat(studentData.firstTermGpa),
                    secondTermGpa: parseFloat(studentData.secondTermGpa),
                    firstLanguage: studentData.firstLanguage,
                    funding: studentData.funding,
                    school: studentData.school,
                    fastTrack: studentData.fastTrack,
                    coop: studentData.coop,
                    residency: studentData.residency,
                    gender: studentData.gender,
                    previousEducation: studentData.previousEducation,
                    ageGroup: studentData.ageGroup,
                    highSchoolAverageMark: parseFloat(studentData.highSchoolAverageMark),
                    mathScore: parseFloat(studentData.mathScore),
                    englishGrade: parseFloat(studentData.englishGrade)
                }
            });

            setPredictionResult(response.data.prediction?.[0]?.[0]); // Adjusted for expected response format
            setError(''); // Clear any previous errors
        } catch (err) {
            setError('Failed to get prediction. Please try again.');
            console.error(err);
        } finally {
            setLoading(false); // stop loading spinner
        }
    };


    return (
        <div className="form-container">
            <div className="form-header-container">
                <img src={cutegradImage} alt="Cute Grad" className="form-header-icon" />
                <h1 className="form-header">Student Persistence Predictor</h1>
            </div>
             {/* body of text*/}
             <div className="form-description">
                <p>
                    This full-stack application helps predict student persistence in academic programs based 
                    on the following key metrics. Simply fill out the form below, and the app will 
                    use machine learning to provide a prediction.
                </p>
            </div>
            <h2>Enter Student Information</h2>
            <form onSubmit={handleSubmit} className="prediction-form">
                {/* all form fields */}
                <div className="form-group">
                    <label>First Term GPA:</label>
                    <input
                        type="number"
                        step="0.1"
                        min="0.0"
                        max="4.5"
                        name="firstTermGpa"
                        value={studentData.firstTermGpa}
                        onChange={handleChange}
                        placeholder="Enter GPA (0.0, 4.5)"
                    />
                </div>
                <div className="form-group">
                    <label>Second Term GPA:</label>
                    <input
                        type="number"
                        step="0.1"
                        min="0.0"
                        max="4.5"
                        name="secondTermGpa"
                        value={studentData.secondTermGpa}
                        onChange={handleChange}
                        placeholder="Enter GPA (0.0, 4.5)"
                    />
                </div>
                <div className="form-group">
                    <label>First Language:</label>
                    <select
                        type="text"
                        name="firstLanguage"
                        value={studentData.firstLanguage}
                        onChange={handleChange}
                    >
                        <option value=""></option>
                        <option value="1">English</option>
                        <option value="2">French</option>
                        <option value="3">Other</option>
                    </select>
                </div>
                <div className="form-group">
                    <label>Co-op:</label>
                    <select
                        name="coop"
                        value={studentData.coop}
                        onChange={handleChange}
                    >
                        <option value=""></option>
                        <option value="1">Yes</option>
                        <option value="2">No</option>
                    </select>
                </div>
                <div className="form-group">
                    <label>Funding:</label>
                    <select
                        type="text"
                        name="funding"
                        value={studentData.funding}
                        onChange={handleChange}
                    >
                        <option value=""></option>
                        <option value="1">Apprentice_PS</option>
                        <option value="2">GPOG_FT</option>
                        <option value="3">Intl OffShore</option>
                        <option value="4">Intl Regular</option>
                        <option value="5">Intl Transfer</option>
                        <option value="6">Joint Program Ryerson</option>
                        <option value="7">Joint Program UTSC</option>
                        <option value="8">Second Career Program</option>
                        <option value="9">Work Safety Insurance Board</option>
                    </select>
                </div>
                <div className="form-group">
                    <label>Residency:</label>
                    <select
                        name="residency"
                        value={studentData.residency}
                        onChange={handleChange}
                    >
                        <option value=""></option>
                        <option value="domestic">Domestic</option>
                        <option value="international">International</option>
                    </select>
                </div>
                <div className="form-group">
                    <label>School:</label>
                    <select
                        type="text"
                        name="school"
                        value={studentData.school}
                        onChange={handleChange}
                    >
                        <option value=""></option>
                        <option value="1">Advancement</option>
                        <option value="2">Business</option>
                        <option value="3">Communications</option>
                        <option value="4">Community and Healt</option>
                        <option value="5">Hospitality</option>
                        <option value="6">Engineering</option>
                        <option value="7">Transportation</option>
                    </select>
                </div>
                <div className="form-group">
                    <label>Gender:</label>
                    <select
                        name="gender"
                        value={studentData.gender}
                        onChange={handleChange}
                    >
                        <option value=""></option>
                        <option value="1">Male</option>
                        <option value="2">Female</option>
                        <option value="3">Neutral</option>
                    </select>
                </div>
                <div className="form-group">
                    <label>Fast Track:</label>
                    <select
                        name="fastTrack"
                        value={studentData.fastTrack}
                        onChange={handleChange}
                    >
                        <option value=""></option>
                        <option value="1">Yes</option>
                        <option value="2">No</option>
                    </select>
                </div>
                <div className="form-group">
                    <label>Previous Education:</label>
                    <select
                        name="previousEducation"
                        value={studentData.previousEducation}
                        onChange={handleChange}
                    >
                        <option value=""></option>
                        <option value="1">High School</option>
                        <option value="2">Post Secondary</option>
                    </select>
                </div>
                <div className="form-group">
                    <label>Age Group:</label>
                    <select
                        name="ageGroup"
                        value={studentData.ageGroup}
                        onChange={handleChange}
                    >
                        <option value=""></option>
                        <option value="1">0-18</option>
                        <option value="2">19-20</option>
                        <option value="3">21-25</option>
                        <option value="4">26-30</option>
                        <option value="5">31-35</option>
                        <option value="6">36-40</option>
                        <option value="7">41-50</option>
                        <option value="8">51-60</option>
                        <option value="9">61-65</option>
                        <option value="10">65+</option>
                    </select>
                </div>
                <div className="form-group">
                    <label>High School Average Mark:</label>
                    <input
                        type="number"
                        step="0.5"
                        min="0.0"
                        max="100.0"
                        name="highSchoolAverageMark"
                        value={studentData.highSchoolAverageMark}
                        onChange={handleChange}
                        placeholder="Enter between [0.0, 100.0]"
                    />
                </div>
                <div className="form-group">
                    <label>Math Grade:</label>
                    <input
                        type="number"
                        step="0.5"
                        min="0.0"
                        max="100.0"
                        name="mathScore"
                        value={studentData.mathScore}
                        onChange={handleChange}
                        placeholder="Enter between [0.0, 50.0]"
                    />
                </div>
                <div className="form-group">
                    <label>English Grade:</label>
                    <select
                        type="number"
                        step="0.1"
                        name="englishGrade"
                        value={studentData.englishGrade}
                        onChange={handleChange}
                    >
                        <option value=""></option>
                        <option value="1">Level-130</option>
                        <option value="2">Level-131</option>
                        <option value="3">Level-140</option>
                        <option value="4">Level-141</option>
                        <option value="5">Level-150</option>
                        <option value="6">Level-151</option>
                        <option value="7">Level-160</option>
                        <option value="8">Level-161</option>
                        <option value="9">Level-170</option>
                        <option value="10">Level-171</option>
                        <option value="11">Level-180</option>
                    </select>
                </div>
                <button type="submit" className="submit-button" disabled={loading}>
                    {loading ? 'Predicting...' : 'Predict'}
                </button>
            </form>


            {validationError && 
                <div className="validation-error">
                    <p>{validationError}</p>
                </div>}
            {predictionResult && (
                <div className="prediction-result">
                    <h3>Prediction Result:</h3>
                    <p>{predictionResult}</p>
                </div>
            )}
            {error && <div className="error-message"><p>{error}</p></div>}
        </div>
    );
};

export default PredictionForm;
