''' The purpose of this file is to show a list of recommended SQL statements for adding, updating, deleting, and selecting data from the database. It then demonstrates how each type of statement can be applied using python. '''

import sqlite3

def QueryTable(query, data = None):
	'''The purpose of this function is to take data stored as a tuple along with a query listed below, to perform that action on the database'''
	conn = sqlite3.connect('Final_db')
	c = conn.cursor()
	# Only pass through no data for select statements, so we want to print the output
	if data == None:
		c.execute(query)
		print('View row: ' + str(c.fetchall()))
		print('')
	else:
		print('input: ' + str(data))
		print('query: ' + query)
		c.execute(query, data)
	conn.commit()
	conn.close()


#The following SQL queries are used to add new rows to a given table in the database.
add_student = "INSERT INTO Student (student_id, programme_id, firstname, surname, dob, email, country_of_domicile, address, language_requirement, academic_qualification) VALUES(?,?,?,?,?,?,?,?,?,?)"
add_grade = "INSERT INTO Grades (grade_id, class_id, assessment_id, student_id, mark) VALUES(?,?,?,?,?)"
add_teacher_notes = "INSERT INTO Teacher_Notes (note_id, class_id, student_id, note) VALUES(?,?,?,?)"
add_assessment = "INSERT INTO Assessments (assessment_id, module_id, assessment_type, assessment_weight, assessment_description) VALUES(?,?,?,?,?)"
add_program_module = "INSERT INTO Program_Modules (programme_id, module_id) VALUES(?,?)"
add_module_enrollment = "INSERT INTO Module_Enrollment (class_id, student_id, final_grade) VALUES(?,?,?)"
add_class = "INSERT INTO Class (class_id, module_id, employee_id, start_date, end_date) VALUES(?,?,?,?,?)"
add_module = "INSERT INTO Modules (module_id, module_name, module_duration, credits, module_description) VALUES(?,?,?,?,?)"
add_programme = "INSERT INTO Programmes (programme_id, programme_name, required_academic_qualification, required_language_qualification, required_credits, cost_per_credit) VALUES(?,?,?,?,?,?)"
add_employee = "INSERT INTO Employee (employee_id, firstname, surname, email) VALUES(?,?,?,?)"



#The following SQL queries are used to update existing rows in a table. These recommended functions show how to update the tables that have allowed NULL values.
update_student = "UPDATE Student SET language_requirement = ? WHERE student_id = ?"
update_module_enrollment = "UPDATE Module_Enrollment SET final_grade = ? WHERE class_id = ? AND student_id = ?"



#The following SQL queries are used to select rows in a table
select_students_missing_data = "SELECT * FROM Student WHERE academic_qualification IS NULL"
select_missing_grades = "SELECT * FROM Module_Enrollment WHERE final_grade IS NULL"
select_student_teacher_module_combinations = "SELECT s.firstname, s.surname, m.module_name, e.firstname, e.surname FROM Student s \
INNER JOIN Module_enrollment me ON s.student_id = me.student_id \
INNER JOIN Class c ON me.class_id = c.class_id \
INNER JOIN Employee e ON c.employee_id = e.employee_id \
INNER JOIN Modules m ON c.module_id = m.module_id \
WHERE s.student_id = 1"



#The following SQL queries are used to delete a student from the database
remove_enrollment = "DELETE FROM Module_Enrollment WHERE student_id = ?"
remove_grade = "DELETE FROM Grades WHERE student_id = ?"
remove_notes = "DELETE FROM Teacher_Notes WHERE student_id = ?"
remove_student = "DELETE FROM Student WHERE student_id = ?"




# Demo for each type of function

#Insert a new student with incomplete data
QueryTable(add_student,(95,1,'Test','Student','1988/07/04','test.student@email.com',None,'72 Fake Road',None,None))
QueryTable(select_students_missing_data)

# Update NULL value from the new entry
QueryTable(update_student, ('English',95))
QueryTable(select_students_missing_data)

# Delete student from student table
QueryTable(remove_student, (95,))
QueryTable(select_students_missing_data)