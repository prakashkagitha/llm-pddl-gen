(define (problem blocksworld-p19)
  (:domain blocksworld)
  (:objects block1 block2 block3 block4 block5 block6 block7 block8 block9 block10 block11 block12 block13 )
  (:init 
    (on-table block1)
    (clear block1)
    (on-table block3)
    (clear block3)
    (on-table block2)
    (clear block2)
    (on-table block10)
    (on block7 block10)
    (on block13 block7)
    (on block5 block13)
    (clear block5)
    (on-table block9)
    (on block8 block9)
    (on block4 block8)
    (on block12 block4)
    (on block11 block12)
    (on block6 block11)
    (clear block6)
    (arm-empty)
  )
  (:goal (and 
    (on-table block7)
    (on-table block6)
    (on-table block4)
    (on-table block9)
    (on-table block10)
    (on-table block12)
    (on-table block11)
    (on-table block8)
    (on-table block3)
    (on-table block5)
    (on-table block13)
    (on-table block2)
    (on-table block1)
  ))
)