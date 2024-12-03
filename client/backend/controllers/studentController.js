const Student = require('../models/Student');

exports.createStudent = async (req, res) => {
    try {
        const { firstTermGpa, secondTermGpa, firstLanguage, funding } = req.body;
        // Input validation
        if (firstTermGpa == null || secondTermGpa == null || firstLanguage == null || funding == null) {
            return res.status(400).json({ error: 'All fields are required' });
        }
        const newStudent = new Student(req.body);
        const savedStudent = await newStudent.save();
        res.status(201).json(savedStudent);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
};

exports.getStudents = async (req, res) => {
    try {
        const students = await Student.find();
        res.status(200).json(students);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
};