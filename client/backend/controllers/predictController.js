const axios = require('axios');

exports.getPrediction = async (req, res) => {
    try {
        const studentData = req.body;
        // Validate student data
        if (!studentData || !studentData.features) {
            return res.status(400).json({ error: 'Invalid input data' });
        }
        // Send student data to Python backend for prediction
        const response = await axios.post('http://localhost:8000/predict', studentData);
        res.json(response.data);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
};