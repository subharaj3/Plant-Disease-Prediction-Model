import express, { type Request, type Response } from 'express';
import cors from 'cors';
import multer from 'multer';
import axios from 'axios';
import FormData from 'form-data';

const app = express();
const PORT = process.env.PORT || 3000;

// FastAPI server URL (Running locally)
const FASTAPI_URL = 'http://localhost:8000/predict/';

// Middleware
app.use(cors());
app.use(express.json());

// Configure Multer to store the uploaded file in memory 
// (We don't need to save it to disk, just forward it to Python)
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// Health check endpoint
app.get('/api/health', (req: Request, res: Response) => {
    res.json({ status: 'Node server is running!' });
});

// Main prediction endpoint
app.post('/api/analyze', upload.single('image'), async (req: Request, res: Response): Promise<void> => {
    try {
        if (!req.file) {
            res.status(400).json({ error: 'No image provided.' });
            return;
        }

        console.log(`Received image: ${req.file.originalname}, forwarding to ML service...`);

        // Create a new form data object to send to FastAPI
        const formData = new FormData();
        
        // Append the file buffer. We must provide the filename and content type
        // so FastAPI's UploadFile dependency recognizes it properly.
        formData.append('file', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype,
        });

        // Make the request to the Python microservice
        const pythonResponse = await axios.post(FASTAPI_URL, formData, {
            headers: {
                ...formData.getHeaders(), // Important: sets the correct multipart boundary
            },
        });

        // Send the prediction back to the client
        res.json({
            success: true,
            data: pythonResponse.data
        });

    } catch (error: any) {
        console.error('Error communicating with ML service:', error.message);
        
        // Check if the error came from the Python server
        if (error.response) {
             res.status(error.response.status).json({ 
                success: false, 
                error: 'ML Service Error', 
                details: error.response.data 
            });
            return;
        }

        res.status(500).json({ success: false, error: 'Internal Server Error' });
    }
});

app.listen(PORT, () => {
    console.log(`Node API Gateway running on http://localhost:${PORT}`);
});