
#!/usr/bin/env python3
#This script begins with a line containing the #! character 
#combination, which is commonly called hash bang or shebang, 
#and continues with the path to the interpreter. If the kernel 
#finds that the first two bytes are #! then it uses the rest 
#of the line as an interpreter and passes the file as an argument. 
#We will use the following shebang in this script:

# Import the CSV module to work with CSV files
import csv

# Define the function read_employees. This function reads employee data from a CSV file.
# It takes the path to the CSV file as a parameter.
def read_employees(csv_file_location):
    # Register a custom CSV dialect to handle potential issues
    csv.register_dialect('empDialect', skipinitialspace=True, strict=True)
    # Open the CSV file and treat it as a dictionary
    employee_file = csv.DictReader(open(csv_file_location), dialect='empDialect')
    employee_list = []  # Create an empty list to store employee data

    # Iterate through each row (employee data) in the CSV file
    for data in employee_file:
        employee_list.append(data)  # Add the employee data to the list

    return employee_list  # Return the list of employee data

# Call the function read_employees with the correct file path
employee_list = read_employees('/home/student-03-6541056b37ef/data/employees.csv')

# Define the function process_data. This function processes employee data to generate
# a dictionary that counts the number of employees in each department.
def process_data(employee_list):
    department_list = []  # Create an empty list to store department names

    # Iterate through each employee's data in the employee_list
    for employee_data in employee_list:
        department_list.append(employee_data['Department'])  # Add the department name to the list

    department_data = {}  # Create an empty dictionary to store department counts
    # Iterate through each unique department name in the department_list
    for department_name in set(department_list):
        # Count the occurrences of the department name and store it in the dictionary
        department_data[department_name] = department_list.count(department_name)

    return department_data  # Return the dictionary containing department counts

# Process the employee_list using the process_data function
dictionary = process_data(employee_list)

# Define the function write_report. This function writes the department data
# dictionary to a report file.
def write_report(dictionary, report_file):
    # Open the report file in write mode ('w+')
    with open(report_file, "w+") as f:
        # Iterate through keys (department names) in the sorted dictionary
        for k in sorted(dictionary):
            # Write department name and count in the format "department:count" to the file
            f.write(str(k) + ':' + str(dictionary[k]) + '\n')

# Call the write_report function to write the department data dictionary to a report file
write_report(dictionary, '/home/student-03-6541056b37ef/test_report.txt')
# Replace '/home/student-03-6541056b37ef/test_report.txt' with the actual file path
