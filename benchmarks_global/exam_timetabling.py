import cpmpy as cp
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar
import json
from datetime import datetime


def day_of_exam(course, slots_per_day):
    return course // slots_per_day


def construct_examtt_simple(nsemesters=9, courses_per_semester=6, slots_per_day=9, days_for_exams=14):
    total_courses = nsemesters * courses_per_semester
    total_slots = slots_per_day * days_for_exams

    parameters = {
        'nsemesters': nsemesters,
        'courses_per_semester': courses_per_semester,
        'slots_per_day': slots_per_day,
        'days_for_exams': days_for_exams
    }

    variables = cp.intvar(1, total_slots, shape=(nsemesters, courses_per_semester), name="var")

    model = cp.Model()
    
    # all exams must be scheduled in different timeslots
    model += cp.AllDifferent(variables)

    # exams in the same semester must be on different days
    for semester_index, row in enumerate(variables):
        exam_days = [day_of_exam(course, slots_per_day) for course in row]
        model += cp.AllDifferent(exam_days)
    
    # limit the number of exams per day
    all_exams = variables.flatten()
    max_exams_per_day = (total_courses + days_for_exams - 1) // days_for_exams + 1
    for day in range(days_for_exams):
        #  how many exams are scheduled on this day
        exams_on_day = cp.Count([day_of_exam(exam, slots_per_day) for exam in all_exams], day)
        # limit the number of exams on this day
        model += (exams_on_day <= max_exams_per_day)

    # Save TRUE constraints for oracle (before adding mocks)
    C_T = list(model.constraints)

    # MOCK OVER-FITTED CONSTRAINTS (will be consistent with 5 examples but NOT generally valid)
    # MUST be global constraints (arity > 3) for Phase 1 detection
    # NOTE: Mocks are added to model for example generation but NOT to oracle (C_T)
    mock_constraints = []

    # # Mock 1: First two semesters cannot share any days (overly restrictive)
    # # TRUE constraint only requires exams WITHIN a semester on different days
    # # This mock requires exams ACROSS first two semesters to use completely different days
    # if nsemesters >= 2:
    #     first_semester_exams = variables[0, :]
    #     second_semester_exams = variables[1, :]
    #     first_semester_days = [day_of_exam(exam, slots_per_day) for exam in first_semester_exams]
    #     second_semester_days = [day_of_exam(exam, slots_per_day) for exam in second_semester_exams]
    #     # Require all days across both semesters to be different (overly restrictive)
    #     mock_c1 = cp.AllDifferent(first_semester_days + second_semester_days)
    #     mock_constraints.append(mock_c1)
    #     model += mock_c1

    # # Mock 2: Middle day must have exactly a specific number of exams (overly specific)
    # # Exact equality is too strict compared to <= constraint
    # if days_for_exams >= 3:
    #     middle_day = days_for_exams // 2
    #     exams_on_middle_day = cp.Count([day_of_exam(exam, slots_per_day) for exam in all_exams], middle_day)
    #     # Require exactly 3 exams on middle day (arbitrary constraint that may hold in examples)
    #     mock_c2 = (exams_on_middle_day == 3)
    #     mock_constraints.append(mock_c2)
    #     model += mock_c2

    AV = absvar(2)  

    lang = [
        AV[0] == AV[1],
        AV[0] != AV[1],
        AV[0] < AV[1],
        AV[0] > AV[1],
        AV[0] >= AV[1],
        AV[0] <= AV[1],
        day_of_exam(AV[0], slots_per_day) != day_of_exam(AV[1], slots_per_day),
        day_of_exam(AV[0], slots_per_day) == day_of_exam(AV[1], slots_per_day),
        cp.Count(AV, AV[0]) <= AV[1], 
        cp.Count(AV, AV[0]) == AV[1] 
    ]

    instance = ProblemInstance(
        variables=variables,
        params=parameters,
        language=lang,
        name=f"exam_timetabling_semesters{nsemesters}_courses{courses_per_semester}_"
             f"timeslots{slots_per_day}_days{days_for_exams}"
    )

    oracle = ConstraintOracle(C_T)
    
    # Return mock constraints so Phase 1 can use them as overfitted constraints
    return instance, oracle, mock_constraints


def generate_exam_timetabling_instance(instance_params=None):
    if instance_params is None:
        instance_params = {}
    
    nsemesters = instance_params.get('nsemesters', 9)
    courses_per_semester = instance_params.get('courses_per_semester', 6)
    slots_per_day = instance_params.get('slots_per_day', 9)
    days_for_exams = instance_params.get('days_for_exams', 14)
    
    return construct_examtt_simple(nsemesters, courses_per_semester, slots_per_day, days_for_exams)


class ExamTimetablingModel:
    def __init__(self, nsemesters=9, courses_per_semester=6, slots_per_day=9, days_for_exams=14, instance_id=1):
        self.nsemesters = nsemesters
        self.courses_per_semester = courses_per_semester
        self.slots_per_day = slots_per_day
        self.days_for_exams = days_for_exams
        self.instance_id = instance_id
        self.total_courses = nsemesters * courses_per_semester
        self.total_slots = slots_per_day * days_for_exams
        
        self.exam_slots = cp.intvar(1, self.total_slots, shape=(nsemesters, courses_per_semester), name="exam_slots")
        
        self.model = cp.Model()
        
        self.add_constraints()
        
        self.solution = None
    
    def add_constraints(self):
        self.model += cp.AllDifferent(self.exam_slots)
        
        for semester_index, row in enumerate(self.exam_slots):
            exam_days = [day_of_exam(course, self.slots_per_day) for course in row]
            self.model += cp.AllDifferent(exam_days)
            
        # Add Count constraint for limiting exams per day
        all_exams = self.exam_slots.flatten()
        total_courses = self.nsemesters * self.courses_per_semester
        max_exams_per_day = (total_courses + self.days_for_exams - 1) // self.days_for_exams + 1
        
        for day in range(self.days_for_exams):
            exams_on_day = cp.Count([day_of_exam(exam, self.slots_per_day) for exam in all_exams], day)
            self.model += (exams_on_day <= max_exams_per_day)
    
    def solve(self):
        if self.model.solve():
            self.create_solution_json()
            return True
        return False
    
    def create_solution_json(self):
        solution = {
            "instance_id": self.instance_id,
            "parameters": {
                "nsemesters": self.nsemesters,
                "courses_per_semester": self.courses_per_semester,
                "slots_per_day": self.slots_per_day,
                "days_for_exams": self.days_for_exams
            },
            "schedule": {}
        }
        
        for semester in range(self.nsemesters):
            semester_name = f"Semester_{semester+1}"
            solution["schedule"][semester_name] = []
            
            for course_idx in range(self.courses_per_semester):
                course_name = f"Course_{semester+1}_{course_idx+1}"
                slot = self.exam_slots[semester, course_idx].value()
                day = day_of_exam(slot, self.slots_per_day) + 1  # 1-indexed day
                time_slot = (slot - 1) % self.slots_per_day + 1  # 1-indexed time slot
                
                solution["schedule"][semester_name].append({
                    "course": course_name,
                    "day": day,
                    "time_slot": time_slot,
                    "slot_id": slot
                })
        
        self.solution = solution


def generate_multiple_solutions(num_solutions=5, nsemesters=9, courses_per_semester=6, 
                               slots_per_day=9, days_for_exams=14):
    all_solutions = []
    
    for i in range(num_solutions):
        current_nsemesters = max(2, nsemesters - 2 + i % 5)  
        current_courses = max(2, courses_per_semester - 1 + i % 3)  
        
        print(f"\nGenerating solution {i+1}/{num_solutions}...")
        print(f"Parameters: {current_nsemesters} semesters, {current_courses} courses per semester")
        
        model = ExamTimetablingModel(
            nsemesters=current_nsemesters,
            courses_per_semester=current_courses,
            slots_per_day=slots_per_day,
            days_for_exams=days_for_exams,
            instance_id=i+1
        )
        
        if model.solve():
            print("Solution found!")
            all_solutions.append(model.solution)
        else:
            print("Failed to find solution.")
    
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_solutions": len(all_solutions),
            "problem_type": "exam_timetabling"
        },
        "solutions": all_solutions
    }
    
    with open("exam_timetabling_solutions.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nGenerated {len(all_solutions)} solutions saved to 'exam_timetabling_solutions.json'")
    return all_solutions


if __name__ == "__main__":
    solutions = generate_multiple_solutions(5)
