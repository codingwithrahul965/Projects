
import numpy as np

# Student names
students = np.array([
    "Rahul",
    "Aman",
    "Priya",
    "Simran",
    "Karan"
])

# Marks in three subjects: Python, Maths, and Computer Science
marks = np.array([
    [85, 90, 88],
    [72, 65, 70],
    [95, 92, 98],
    [40, 55, 48],
    [30, 35, 25]
])

subjects = ["Python", "Maths", "Computer Science"]

# Calculate total and percentage for each student
total_marks = np.sum(marks, axis=1)
percentage = np.mean(marks, axis=1)

# Calculate subject-wise averages
subject_averages = np.mean(marks, axis=0)

# Find the highest and lowest individual marks
highest_mark = np.max(marks)
lowest_mark = np.min(marks)

# Find the student with the highest total
topper_index = np.argmax(total_marks)

print("=" * 45)
print("       STUDENT PERFORMANCE ANALYZER")
print("=" * 45)

for i in range(len(students)):
    result = "PASS" if np.all(marks[i] >= 40) else "FAIL"

    print(f"\nStudent: {students[i]}")
    print(f"Marks: {marks[i]}")
    print(f"Total: {total_marks[i]} / 300")
    print(f"Percentage: {percentage[i]:.2f}%")
    print(f"Result: {result}")

print("\n" + "=" * 45)
print("          CLASS SUMMARY")
print("=" * 45)

for i in range(len(subjects)):
    print(f"{subjects[i]} Average: {subject_averages[i]:.2f}")

print(f"\nHighest individual mark: {highest_mark}")
print(f"Lowest individual mark: {lowest_mark}")
print(f"Class average: {np.mean(marks):.2f}")
print(f"Topper: {students[topper_index]}")
print(f"Topper's total: {total_marks[topper_index]} / 300")
