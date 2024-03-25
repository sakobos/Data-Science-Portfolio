# Read in data and select only Coastal Carolina Pitchers ("COA_CHA")
> MasterTrackman= read.delim("clipboard", header= TRUE)
> attach(MasterTrackman)
> CoastalTotal= subset(MasterTrackman, PitcherTeam == "COA_CHA" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )

# Filtering out 4 Pitchers that throw from a "submarine" arm slot
> CoastalDrop1= subset( CoastalTotal, Pitcher != "Orlando, Patrick" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalDrop2= subset( CoastalDrop1, Pitcher != "Inman, David" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows","TaggedPitchType") )
> CoastalDrop3= subset( CoastalDrop2, Pitcher != "Abney, Alaska" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows","TaggedPitchType") )
> CoastalPitchers=  subset( CoastalDrop3, Pitcher != "Beckwith, Andrew" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )

# Separate the dataset into Left-Handed and Right-Handed Coastal Carolina Pitchers (LHP = Left-Handed Pitcher, RHP= Right_Handed Pitcher)
> CoastalLefties= subset( CoastalPitchers, PitcherThrows == "Left" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties= subset( CoastalPitchers, PitcherThrows == "Right" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )

# Selecting only the Fastball pitch type for each group
 > CoastalLeftiesFB= subset( CoastalLefties, TaggedPitchType == "Fastball" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
 > CoastalRightiesFB= subset( CoastalRighties, TaggedPitchType == "Fastball" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )

# Getting only LHP 4-seam Fastballs by removing pitches with Tilt-metric that do not match 4-seam Fastballs. Now CoastalLHP4S 
> CoastalLeftiesFB1= subset( CoastalLeftiesFB, Tilt != "9:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLeftiesFB2= subset( CoastalLeftiesFB1, Tilt != "10:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLeftiesFB3= subset( CoastalLeftiesFB2, Tilt != "10:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLeftiesFB4= subset( CoastalLeftiesFB3, Tilt != "10:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLeftiesFB5= subset( CoastalLeftiesFB4, Tilt != "10:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLeftiesFB6= subset( CoastalLeftiesFB5, Tilt != "12:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties7= subset( CoastalLeftiesFB6, Tilt != "12:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties8= subset( CoastalLefties7, Tilt != "12:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties9= subset( CoastalLefties8, Tilt != "1:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties10= subset( CoastalLefties9, Tilt != "1:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties11= subset( CoastalLefties10, Tilt != "1:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties12= subset( CoastalLefties11, Tilt != "1:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties13= subset( CoastalLefties12, Tilt != "2:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties14= subset( CoastalLefties13, Tilt != "9:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLHP4S= subset( CoastalLefties14, Tilt != "2:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )

# Getting only LHP 2-seam Fastballs by removing pitches with Tilt-metric that do not match 2-seam Fastballs. Now CoastalLHP2S 
> CoastalLefties2Seam1= subset( CoastalLeftiesFB, Tilt != "12:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties2Seam2= subset( CoastalLefties2Seam1, Tilt != "11:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties2Seam3= subset( CoastalLefties2Seam2, Tilt != "11:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties2Seam4= subset( CoastalLefties2Seam3, Tilt != "11:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties2Seam5= subset( CoastalLefties2Seam4, Tilt != "11:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties2Seam6= subset( CoastalLefties2Seam5, Tilt != "9:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties2Seam7= subset( CoastalLefties2Seam6, Tilt != "9:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties2Seam8= subset( CoastalLefties2Seam7, Tilt != "12:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties2Seam9= subset( CoastalLefties2Seam8, Tilt != "12:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties2Seam10= subset( CoastalLefties2Seam9, Tilt != "12:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties2Seam11= subset( CoastalLefties2Seam10, Tilt != "1:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties2Seam12= subset( CoastalLefties2Seam11, Tilt != "1:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties2Seam13= subset( CoastalLefties2Seam12, Tilt != "1:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLefties2Seam14= subset( CoastalLefties2Seam13, Tilt != "1:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLHP15= subset( CoastalLefties2Seam14, Tilt != "2:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed" "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLHP16= subset( CoastalLHP15, Tilt != "2:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLHP2s= subset( CoastalLHP16, Tilt != "2:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalLHP2S= subset( CoastalLHP2s, Tilt != "2:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )

# Getting only RHP 4-seam Fastballs by removing pitches with Tilt-metric that do not match 4-seam Fastballs. Now CoastalRHP4S 
> CoastalRighties4S1= subset( CoastalRightiesFB, Tilt != "1:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S2= subset( CoastalRighties4S1, Tilt != "1:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S3= subset( CoastalRighties4S2, Tilt != "1:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S4= subset( CoastalRighties4S3, Tilt != "2:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S5= subset( CoastalRighties4S4, Tilt != "2:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S6= subset( CoastalRighties4S5, Tilt != "2:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S7= subset( CoastalRighties4S6, Tilt != "2:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S8= subset( CoastalRighties4S7, Tilt != "3:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S9= subset( CoastalRighties4S8, Tilt != "11:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S10= subset( CoastalRighties4S9, Tilt != "11:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S11= subset( CoastalRighties4S10, Tilt != "11:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S12= subset( CoastalRighties4S11, Tilt != "11:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S13= subset( CoastalRighties4S12, Tilt != "10:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S14= subset( CoastalRighties4S13, Tilt != "10:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S15= subset( CoastalRighties4S14, Tilt != "10:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S16= subset( CoastalRighties4S15, Tilt != "10:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S17= subset( CoastalRighties4S16, Tilt != "9:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S18= subset( CoastalRighties4S17, Tilt != "9:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties4S19= subset( CoastalRighties4S18, Tilt != "9:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRHP4S= subset( CoastalRighties4S19, Tilt != "9:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )

# Getting only RHP 2-seam Fastballs by removing pitches with Tilt-metric that do not match 2-seam Fastballs. Now CoastalRHP2S 
> CoastalRighties2S1= subset( CoastalRightiesFB, Tilt != "9:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S2= subset( CoastalRighties2S1, Tilt != "9:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S3= subset( CoastalRighties2S2, Tilt != "9:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S4= subset( CoastalRighties2S3, Tilt != "9:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S5= subset( CoastalRighties2S4, Tilt != "10:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S6= subset( CoastalRighties2S5, Tilt != "10:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S7= subset( CoastalRighties2S6, Tilt != "10:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S8= subset( CoastalRighties2S7, Tilt != "10:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S9= subset( CoastalRighties2S8, Tilt != "11:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S10= subset( CoastalRighties2S9, Tilt != "11:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S11= subset( CoastalRighties2S10, Tilt != "11:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S12= subset( CoastalRighties2S11, Tilt != "11:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S13= subset( CoastalRighties2S12, Tilt != "12:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S14= subset( CoastalRighties2S13, Tilt != "12:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S15= subset( CoastalRighties2S14, Tilt != "12:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S16= subset( CoastalRighties2S15, Tilt != "12:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S17= subset( CoastalRighties2S16, Tilt != "1:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S18= subset( CoastalRighties2S17, Tilt != "1:15" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S19= subset( CoastalRighties2S18, Tilt != "1:30" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRighties2S20= subset( CoastalRighties2S19, Tilt != "2:45" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )
> CoastalRHP2S= subset( CoastalRighties2S20, Tilt != "3:00" , select= c("Tilt","RelHeight", "InducedVertBreak","HorzBreak", "ExitSpeed", "Pitcher", "PitcherThrows", "TaggedPitchType") )


# Using "astroFns" to change the Tilt metric from a "clock" axis to radians to be used in regression
> install.packages("astroFns")
trying URL 'https://cran.rstudio.com/bin/windows/contrib/3.5/astroFns_4.1-0.zip'
Content type 'application/zip' length 72348 bytes (70 KB)
downloaded 70 KB

# Applying hms2rad to change the Tilt metric
> library(astroFns)
> testtilt= hms2rad(Tilt)
> attach(CoastalLHP4S)
The following objects are masked from CoastalLHP2S:

    ExitSpeed, HorzBreak, InducedVertBreak, Pitcher, PitcherThrows, RelHeight, TaggedPitchType, Tilt

The following objects are masked from CoastalLHP4S (pos = 5):

    ExitSpeed, HorzBreak, InducedVertBreak, Pitcher, PitcherThrows, RelHeight, TaggedPitchType, Tilt

> CoastalLHP4STiltFix= hms2rad(Tilt)
> CoastalLHP4STiltFix


# A lesson in project management to be had. I am missing the remainder of the code that applied the hms2rad function to the 3 other (LHP2S, RHP4S, RHP2S) datasets. 


