from aitor.aitorflows import Aitorflow
from aitor.task import task

# Define tasks using the decorator
@task
def task1(x):
    print(f"Task 1 processing: {x}")
    return x * 2

@task
def task2(x):
    print(f"Task 2 processing: {x}")
    return x + 10

@task
def task3(x):
    print(f"Task 3 processing: {x}")
    return x * 3

@task
def task4(x, y):
    print(f"Task 4 processing inputs: {x}, {y}")
    return x + y

@task
def task5(x):
    print(f"Task 5 processing: {x}")
    return x - 5

# Create workflow
workflow = Aitorflow(name="Example Workflow")

# Add tasks and define dependencies
task1 >> [task2, task3]
task2 >> task4
task3 >> [task4, task5]

# Add any task to the workflow (it will discover connected tasks)
workflow.add_task(task1)

# Validate workflow
workflow.validate()
workflow.print()

# Execute workflow
results = workflow.execute(initial_input=5)
print("Results:", results) 