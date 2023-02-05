cd body
python body_detection.py ../messi.jpg
TIMEOUT 2
cd ..
cd age
python age_detection.py ../predicted_age.jpg
TIMEOUT 2
cd ..
cd gender
python gender_detection.py ../predicted_age.jpg
pause