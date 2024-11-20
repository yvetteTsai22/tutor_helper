This is a tutor assistant powered by LLM.
The goal is to create a tool that can support a tutor in managing their teaching tasks, enhance student engagement, and streamline administrative tasks. 
### Functions

- **Answering Questions**: The assistant should be able to answer frequently asked questions related to the subject being taught. This could involve automated explanations, quick facts, or resources.
- **Provide Explanations**: Help tutors explain complex concepts to students in simpler terms. For example, the assistant could break down difficult concepts or suggest alternative ways of explaining a topic.
- **Study Plans**: Create personalized study plans for students, adapting to their learning pace and weaknesses. This can be based on their previous performance or goals set by the tutor.
- **Automated Feedback**: After each tutoring session or assignment, the assistant can generate automated feedback for the student based on their performance, including suggestions for improvement.
- **Session Summaries**: Generate and send summaries of tutoring sessions to both tutors and students, outlining what was covered, what needs further review, and next steps.
- **Customized Search**: enhance the search term and search multiple resources at the same time

   ![Customized Search Demo](image/search.gif)


### Installation
#### Prerequisite
- OpenAI key
- Python >= 3.12
- [Optional] Docker environment

1. clone the repository
   ```
   git clone git@github.com:yvetteTsai22/tutor_helper.git
   ```
2. create .env
   ```
   BACKEND_URL=127.0.0.1:8000
   OPENAI_API_KEY=<YOUR KEY></YOUR>
   ```

#### local
3. install packages
   ```
   pipenv shell
   pipenv install -r requirements.txt
   ```
4. start fastAPI
   ```
   cd tutor_helper/
   uvicorn tutor_helper.use_cases.fastapi:app --reload
   ```
5. start strealit
   ```
   streamlit run tutor_helper/use_cases/streamlit_app.py
   ```



#### docker
3. buid docker image
   ```
   docker image build --platform linux/amd64 -f Dockerfile -t tutor_helper-dev:"$TIMESTAMP" .
   ```
4. run docker container
   ```
   docker run -p 8501:8501 -p 8000:8000 tutor_helper-dev:"$TIMESTAMP"
   ```

You should see the app in http://localhost:8501/.