# This folder is specifically designed to run few-shot experiments w/ SDPO

* filter high-quality data based on reward scores
* receive feedbacks from multiple teachers
* these feedbacks are accompanied with environment feedbacks
* Also we may have two ways of receiving feedbacks from teacher model: (1) combing textual feedbacks in the teacher prompt (2) combing log probs as rewards
* Baselines:
    1. teacher/student model before training
    2. teacher/student model RL
    3. SDPO w/ env feedback