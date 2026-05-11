import math

SKILL_ALIASES = {
    # Languages
    "python": "python", "pyhton": "python",
    "java": "java",
    "javascript": "javascript", "javascrpit": "javascript", "js": "javascript",
    "typescript": "typescript", "typescrpit": "typescript",
    "c++": "cpp", "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",
    # ML / Data
    "machinelearning": "machinelearning", "machine learning": "machinelearning",
    "ml": "machinelearning", "sklearn": "machinelearning",
    "deeplearning": "deeplearning", "deep learning": "deeplearning", "deep-learning": "deep_learning",
    "tensorflow": "tensorflow", "pytorch": "pytorch", "keras": "keras",
    "nlp": "nlp", "bert": "bert", "xgboost": "xgboost",
    "feature engineering": "feature_engineering",
    "statistics": "statistics", "stats": "statistics",
    "regression": "regression", "clustering": "clustering",
    "data-viz": "datavisualization", "data visualization": "datavisualization",
    "data viz": "datavisualization", "matplotlib": "datavisualization",
    "tableau": "datavisualization", "power-bi": "datavisualization",
    "power bi": "datavisualization", "powerbi": "datavisualization",
    "pandas": "pandas", "numpy": "numpy",
    # Web — Frontend
    "react": "react", "reacts": "react", "reactjs": "react",
    "vue": "vue", "vue.js": "vue", "vuejs": "vue",
    "redux": "redux", "tailwind": "tailwind",
    "html/css": "htmlcss", "html css": "htmlcss", "html": "htmlcss", "css": "htmlcss",
    "jest": "jest", "graphql": "graphql",
    # Web — Backend
    "node.js": "nodejs", "nodejs": "nodejs", "node js": "nodejs",
    "flask": "flask",
    "spring boot": "springboot", "springboot": "springboot",
    "rest api": "restapi", "rest": "restapi", "restapi": "rest_api",
    "microservices": "microservices",
    # Databases
    "sql": "sql", "mysql": "mysql", "mysq": "mysql",
    "postgresql": "postgresql", "postgres": "postgresql",
    "mongodb": "mongodb", "redis": "redis",
    # DevOps / Cloud
    "docker": "docker",
    "kubernetes": "kubernetes", "kubernates": "kubernetes", "k8s": "kubernetes",
    "ci/cd": "cicd", "cicd": "cicd", "ci cd": "ci_cd",
    "aws": "aws",
    # Mobile
    "android": "android", "firebase": "firebase",
    # CS Fundamentals
    "algorithms": "algorithms", "algoritms": "algorithms",
    "data structure": "datastructures", "data structures": "datastructures",
    "competitive programming": "competitive_programming",
    # Design
    "ui/ux": "uiux", "ui ux": "uiux", "figma": "figma",
}


RESUME_DATASET = [
    {"id": "01", "candidate": "Arjun Sharma",   "raw_skills": "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"},
    {"id": "02", "candidate": "Priya Nair",     "raw_skills": "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"},
    {"id": "03", "candidate": "Rahul Gupta",    "raw_skills": "Java, Spring Boot, MySql, Microservices, Docker, kubernates"},
    {"id": "04", "candidate": "Sneha Patel",    "raw_skills": "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"},
    {"id": "05", "candidate": "Vikram Singh",   "raw_skills": "C++, Algoritms, Data Structure, competitive programming, python"},
    {"id": "06", "candidate": "Ananya Krishnan","raw_skills": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"},
    {"id": "07", "candidate": "Karan Mehta",    "raw_skills": "Python, Sklearn, XGboost, feature engineering, SQL, tableau"},
    {"id": "08", "candidate": "Deepika Rao",    "raw_skills": "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"},
    {"id": "09", "candidate": "Aditya Kumar",   "raw_skills": "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"},
    {"id": "10", "candidate": "Meera Iyer",     "raw_skills": "python, R, statistics, ML, regression, clustering, Power-BI"},
]

JD_DATASET = [
    {"id": "JD-1", "company": "Kakao", "role": "ML Engineer",
     "requiredskills": ["Python","Machine Learning","Deep Learning","TensorFlow","PyTorch","SQL","Data Visualization"],
     "preferredskills": ["NLP","BERT","Feature Engineering","Statistics"]},
    {"id": "JD-2", "company": "Naver", "role": "Backend Engineer",
     "requiredskills": ["Java","Spring Boot","MySQL","PostgreSQL","Microservices","Docker","Kubernetes"],
     "preferredskills": ["REST API","CI/CD","Redis"]},
    {"id": "JD-3", "company": "Line", "role": "Frontend Engineer",
     "requiredskills": ["JavaScript","React","Vue","TypeScript","REST API","HTML/CSS"],
     "preferredskills": ["Node.js","GraphQL","Redux","Jest","AWS"]},
]

class ResumeMatcher:
    def __init__(self, resumedataset, jddataset, skill_aliases):
        self.resumedataset = resumedataset
        self.jddataset = jddataset
        self.skill_aliases = skill_aliases
        self.normalized_resumes = self.normalizeresumes()
        self.vocabulary = self.buildvocabulary()
        self.dftable = self.builddftable()

    def normalizeresumes(self):
        """Normalize and deduplicate skills in resumes."""
        normalized_resumes = {}
        for resume in self.resumedataset:
            rawskills = resume["raw_skills"]
            tokens = [t.strip() for t in rawskills.lower().split(",")]
            seen = set()
            result = []
            for token in tokens:
                canonical = self.skill_aliases.get(token)   # direct lookup only — no substring fallback
                if canonical and canonical not in seen:
                    seen.add(canonical)
                    result.append(canonical)
            normalized_resumes[resume["id"]] = result
        return normalized_resumes

    def buildvocabulary(self):
        """Build shared vocabulary from normalized resumes."""
        all_skills = set()
        for skills in self.normalized_resumes.values():
            all_skills.update(skills)
        return sorted(all_skills)   # alphabetically sorted, consistent ordering

    def builddftable(self):
        """Build document-frequency table."""
        df_table = {
            skill: sum(1 for skills in self.normalized_resumes.values() if skill in skills)
            for skill in self.vocabulary
        }
        return df_table

    def computetfidf(self, skills):
        """Compute TF-IDF vector for a resume."""
        n = len(skills)
        return {skill: (1 / n) * math.log(10 / self.dftable[skill]) for skill in skills}

    def buildjd_vector(self, jd):
        """Build JD binary vector over the shared vocabulary."""
        vector = {}
        for skill_str in jd["requiredskills"] + jd["preferredskills"]:
            canonical = self.skill_aliases.get(skill_str.lower().strip())
            if canonical and canonical in self.vocabulary:   # must be in shared vocab
                vector[canonical] = 1
        return vector

    def cosinesimilarity(self, tfidf, jd_vec):
        """Compute cosine similarity between TF-IDF and JD vectors."""
        dot      = sum(tfidf.get(s, 0) * jd_vec.get(s, 0) for s in set(tfidf) | set(jd_vec))
        mag_a    = math.sqrt(sum(v ** 2 for v in tfidf.values()))
        magb    = math.sqrt(sum(v **  2 for v in jd_vec.values()))
        if mag_a == 0 or magb == 0:
            return 0.0
        return dot / (mag_a * magb)

    def rank_resumes(self):
        """Rank and print top-3 resumes per JD."""
        for jd in self.jddataset:
            jdvec = self.buildjd_vector(jd)
            scores = []
            for resume in self.resumedataset:
                skills  = self.normalized_resumes[resume["id"]]
                tfidf   = self.computetfidf(skills)
                sim     = self.cosinesimilarity(tfidf, jdvec)
                scores.append((resume["candidate"], round(sim, 2)))

            # Sort: score descending, then name alphabetically for ties
            scores.sort(key=lambda x: (-x[1], x[0]))
            top3 = scores[:3]

            print(f"{jd['id']} — {jd['company']} ({jd['role']})")
            print(", ".join(f"{name}({score:.2f})" for name, score in top3))
            print()

matcher = ResumeMatcher(RESUME_DATASET, JD_DATASET, SKILL_ALIASES)
matcher.rank_resumes()
