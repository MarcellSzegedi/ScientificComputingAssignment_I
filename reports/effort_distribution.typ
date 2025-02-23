= Team effort distribution report

#show link: set text(fill: blue)

*Assignment number:* 1

*Assignment team number:* 12

*GitHub URL:* #link("https://github.com/MarcellSzegedi/ScientificComputingAssignment_I")

At the start of the assignment, we met to discuss the theory underlying the core of the assignment. Once 
we had developed a shared understanding of the content, and decided on a rough approach for each section,
we divided the work. Zainab and Henry primarily worked on the vibrating string and time-dependent diffusion
questions, while Marcell focused on the time-independent diffusion problems. 

While much of the code was completed individually, we ensured that everyone was kept up-to-date through 
regular meetings and pull requests. At times, we practiced peer-programming, working in small groups on
a single laptop. We all contributed equally to the report.


= Git Fame distribution of the repository
Date and time: #json("effort_distribution.json").datetime

Output from Git Fame:

#let git_fame_summary = csv("git_fame_summary.csv")
#let git_fame_details = csv("git_fame_detailed.csv", )

#show table.cell.where(y: 0): strong
//#figure(
#table(
		columns: 4,
		..git_fame_summary.flatten(),
	)
//)

//#figure(
#table(
		columns: 7,
		..git_fame_details.flatten(),
	)
//)
