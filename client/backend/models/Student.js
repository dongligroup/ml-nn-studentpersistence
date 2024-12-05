const mongoose = require('mongoose');

const StudentSchema = new mongoose.Schema({
    firstTermGpa: {
        type: Number,
        required: true,
        min: 0,
        max: 4.5
    },
    secondTermGpa: {
        type: Number,
        required: true,
        min: 0,
        max: 4.5
    },
    firstLanguage: {
        type: Number,
        required: true,
    },
    funding: {
        type: Number,
        required: true,
    },
    // Add other fields from dataset here
}, { timestamps: true });

module.exports = mongoose.model('Student', StudentSchema);